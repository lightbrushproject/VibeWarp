"""Audio-reactive scheduling.

The audio twin of content-aware scene scheduling (core/scene_detect.py):
where scene_detect rewrites schedules from a per-frame *visual* difference
signal, this module rewrites the same four schedules from a per-frame
*audio* signal, so the music itself drives the render.

Pipeline: decode audio (ffmpeg, stdlib wave fallback) -> per-video-frame
features (RMS energy, sub-250 Hz bass energy, spectral-flux onset strength)
-> compile_schedules maps features onto schedule ranges with smoothing,
curves and onset punches -> apply_audio_schedules writes the results onto
the run config, exactly like scene_detect.apply_content_schedules.

Design constraints:
- CPU / numpy only. No torch, no CUDA, no new dependencies (no librosa):
  ffmpeg is already a hard requirement of the whole project.
- Opt-in and inert by default. Nothing here runs unless config.audio.enabled
  (pipeline) or the tools/audio_schedule.py CLI is invoked directly.
- Fail soft at the pipeline boundary: the pipeline catches audio errors,
  warns, and renders without audio schedules rather than crashing a render.
"""

import os
import shutil
import subprocess
import wave
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_SAMPLE_RATE = 22050
BASS_CUTOFF_HZ = 250.0

# The Lightbrush preset. Rationale: loud passages push style strength and CFG
# up (the image leans harder into the prompt when the music does); bass drives
# warp inheritance; a detected onset punches flow_blend to 0 for one frame
# (fresh diffusion on the hit, like a scene cut) and decays back over 4 frames;
# onsets also raise step count so hit frames render cleaner.
DEFAULT_MAPPING: Dict[str, Dict[str, Any]] = {
    'style_strength_schedule': {
        'source': 'energy', 'min': 0.45, 'max': 0.85,
        'curve': 'linear', 'smooth': 3,
    },
    'cfg_scale_schedule': {
        'source': 'energy', 'min': 5.0, 'max': 9.0,
        'curve': 'linear', 'smooth': 3,
    },
    'flow_blend_schedule': {
        'source': 'bass', 'min': 0.35, 'max': 0.8,
        'curve': 'linear', 'smooth': 2,
        'onset_punch': {'value': 0.0, 'decay_frames': 4},
    },
    'steps_schedule': {
        'source': 'onset', 'min': 15, 'max': 30,
        'curve': 'linear', 'smooth': 0, 'quantize': 1,
    },
}

SCHEDULE_TARGETS = {
    'steps_schedule': ('diffusion', 'steps_schedule'),
    'cfg_scale_schedule': ('diffusion', 'cfg_scale_schedule'),
    'style_strength_schedule': ('diffusion', 'style_strength_schedule'),
    'flow_blend_schedule': ('warp', 'flow_blend_schedule'),
}


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def _decode_with_ffmpeg(path: str, sample_rate: int) -> np.ndarray:
    """Decode any audio/video file to mono float32 via ffmpeg."""
    cmd = [
        'ffmpeg', '-v', 'error', '-i', path,
        '-vn', '-ac', '1', '-ar', str(sample_rate),
        '-f', 'f32le', '-acodec', 'pcm_f32le', '-',
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg could not decode {path!r}: "
            f"{proc.stderr.decode('utf-8', 'replace').strip()[:300]}")
    samples = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if samples.size == 0:
        raise RuntimeError(f"ffmpeg decoded no audio samples from {path!r}")
    return samples


def _decode_wav_stdlib(path: str, sample_rate: int) -> Tuple[np.ndarray, int]:
    """Decode a PCM .wav with the stdlib wave module (ffmpeg fallback)."""
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError(f"unsupported wav sample width: {sampwidth} bytes")
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    if sr != sample_rate and data.size > 1:
        # Linear resample; good enough for feature extraction.
        target = max(1, int(round(data.size * sample_rate / sr)))
        data = np.interp(
            np.linspace(0, data.size - 1, target),
            np.arange(data.size), data).astype(np.float32)
        sr = sample_rate
    return data.astype(np.float32), sr


def decode_audio(path: str, sample_rate: int = DEFAULT_SAMPLE_RATE
                 ) -> Tuple[np.ndarray, int]:
    """Decode an audio (or video) file to mono float32 samples in [-1, 1].

    Uses ffmpeg when available (it handles every container, and extracts the
    audio track from video files). Falls back to the stdlib wave module for
    .wav files when ffmpeg is missing. Raises RuntimeError when neither path
    works.
    """
    if not os.path.isfile(path):
        raise RuntimeError(f"audio file not found: {path!r}")
    if shutil.which('ffmpeg'):
        return _decode_with_ffmpeg(path, sample_rate), sample_rate
    if path.lower().endswith('.wav'):
        return _decode_wav_stdlib(path, sample_rate)
    raise RuntimeError(
        f"cannot decode {path!r}: ffmpeg is not on PATH and the stdlib "
        "fallback only handles .wav files")


# ---------------------------------------------------------------------------
# Per-frame features
# ---------------------------------------------------------------------------

def _robust_normalize(values: np.ndarray) -> np.ndarray:
    """Scale to 0..1 using 5th/99th percentiles; constant input maps to 0."""
    if values.size == 0:
        return values.astype(np.float64)
    lo = np.percentile(values, 5)
    hi = np.percentile(values, 99)
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def frame_features(samples: np.ndarray, sr: int, fps: float,
                   num_frames: int) -> Dict[str, List[float]]:
    """Per-video-frame audio features, each a list of length num_frames.

    Window i covers samples [i*sr/fps, (i+1)*sr/fps). If the audio is shorter
    than num_frames at this fps, the remaining frames repeat the last
    window's values (energy/bass) and get zero onset, so schedules stay
    defined for the whole render.
    """
    samples = np.asarray(samples, dtype=np.float64).ravel()
    hop = max(1, int(round(sr / float(fps))))
    n_windows = int(np.ceil(samples.size / hop)) if samples.size else 0

    energy = np.zeros(max(n_windows, 1))
    bass = np.zeros(max(n_windows, 1))
    flux = np.zeros(max(n_windows, 1))
    prev_mag = None
    for i in range(n_windows):
        window = samples[i * hop:(i + 1) * hop]
        if window.size == 0:
            continue
        energy[i] = np.sqrt(np.mean(window ** 2))
        spectrum = np.abs(np.fft.rfft(window))
        freqs = np.fft.rfftfreq(window.size, d=1.0 / sr)
        bass_band = spectrum[freqs < BASS_CUTOFF_HZ]
        total = spectrum.sum()
        bass[i] = (bass_band.sum() / total) if total > 1e-12 else 0.0
        if prev_mag is not None:
            size = min(prev_mag.size, spectrum.size)
            diff = spectrum[:size] - prev_mag[:size]
            flux[i] = float(np.sum(diff[diff > 0]))
        prev_mag = spectrum

    def _fit(values: np.ndarray) -> List[float]:
        normed = _robust_normalize(values)
        out = np.zeros(num_frames)
        take = min(num_frames, normed.size)
        if take:
            out[:take] = normed[:take]
            if take < num_frames:
                out[take:] = normed[take - 1]
        return [round(float(v), 4) for v in out]

    onset = _fit(flux)
    if num_frames > n_windows and n_windows > 0:
        onset[n_windows:] = [0.0] * (num_frames - n_windows)
    return {'energy': _fit(energy), 'bass': _fit(bass), 'onset': onset}


def detect_onsets(onset_curve: List[float], threshold_percentile: float = 90.0,
                  min_gap_frames: int = 3) -> List[int]:
    """Frame indices of onsets: local peaks above a percentile threshold."""
    curve = np.asarray(onset_curve, dtype=np.float64)
    if curve.size == 0 or curve.max() <= 0:
        return []
    # Floor the threshold at 10% of the peak: on sparse curves (long silences
    # between hits) the percentile itself is 0, which would otherwise make
    # every silent frame a "peak" at threshold 0.
    thresh = max(float(np.percentile(curve, threshold_percentile)),
                 0.1 * float(curve.max()))
    onsets: List[int] = []
    last = -min_gap_frames - 1
    for i, value in enumerate(curve):
        left = curve[i - 1] if i > 0 else -np.inf
        right = curve[i + 1] if i < curve.size - 1 else -np.inf
        if value >= thresh and value >= left and value >= right \
                and i - last > min_gap_frames:
            onsets.append(i)
            last = i
    return onsets


def estimate_tempo(onset_curve: List[float], fps: float) -> Optional[float]:
    """Estimate tempo in BPM by autocorrelating the onset curve.

    Searches lags corresponding to 40..240 BPM and returns the strongest, or
    None when the curve has no clear periodicity. This is a simple estimate,
    not a learned beat tracker.
    """
    curve = np.asarray(onset_curve, dtype=np.float64)
    if curve.size < 8 or curve.max() <= 0:
        return None
    curve = curve - curve.mean()
    min_lag = max(1, int(round(fps * 60.0 / 240.0)))
    max_lag = min(curve.size - 1, int(round(fps * 60.0 / 40.0)))
    if max_lag <= min_lag:
        return None
    corr = np.correlate(curve, curve, mode='full')[curve.size - 1:]
    segment = corr[min_lag:max_lag + 1]
    if segment.size == 0 or segment.max() <= 0:
        return None
    best_lag = min_lag + int(np.argmax(segment))
    # Require real periodicity: normalized autocorrelation at the best lag
    # (relative to lag 0) must clear 0.25, which random onset noise does not.
    if corr[0] <= 0 or corr[best_lag] / corr[0] < 0.25:
        return None
    return round(60.0 * fps / best_lag, 1)


# ---------------------------------------------------------------------------
# Schedule compilation
# ---------------------------------------------------------------------------

def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or values.size == 0:
        return values
    kernel = np.ones(width) / width
    padded = np.pad(values, (width // 2, width - 1 - width // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:values.size]


def _apply_curve(values: np.ndarray, curve: str) -> np.ndarray:
    if curve == 'pow2':
        return values ** 2
    if curve == 'sqrt':
        return np.sqrt(np.clip(values, 0.0, None))
    return values  # 'linear'


def _apply_onset_punch(values: np.ndarray, onsets: List[int],
                       punch: Dict[str, Any]) -> np.ndarray:
    """Set output to punch value at each onset, decaying back linearly."""
    punch_value = float(punch.get('value', 0.0))
    decay = int(punch.get('decay_frames', 0))
    out = values.copy()
    for onset in onsets:
        out[onset] = punch_value
        for j in range(1, decay + 1):
            idx = onset + j
            if idx >= out.size:
                break
            recovered = values[idx]
            out[idx] = punch_value * (decay - j) / decay \
                + recovered * j / decay if decay else recovered
    return out


def compile_schedules(features: Dict[str, List[float]], fps: float,
                      mappings: Optional[Dict[str, Dict[str, Any]]] = None,
                      onset_threshold_percentile: float = 90.0
                      ) -> Dict[str, list]:
    """Compile per-frame schedules from audio features.

    mappings maps a schedule field name to a spec dict (see module docstring
    and DEFAULT_MAPPING). None uses DEFAULT_MAPPING. Returns a dict of
    schedule field name -> per-frame list, rounded to 3 decimals, with
    steps_schedule values as ints.
    """
    if mappings is None:
        mappings = DEFAULT_MAPPING
    if not features or 'energy' not in features:
        raise ValueError("features must come from frame_features()")
    num_frames = len(features['energy'])
    onsets = detect_onsets(features.get('onset', []),
                           threshold_percentile=onset_threshold_percentile)

    schedules: Dict[str, list] = {}
    for name, spec in mappings.items():
        if name not in SCHEDULE_TARGETS:
            raise ValueError(f"unknown schedule target: {name!r} "
                             f"(expected one of {sorted(SCHEDULE_TARGETS)})")
        source = spec.get('source', 'energy')
        if source not in features:
            raise ValueError(f"mapping for {name} uses unknown feature "
                             f"{source!r}")
        curve_vals = _apply_curve(
            np.asarray(features[source], dtype=np.float64),
            spec.get('curve', 'linear'))
        smooth = int(spec.get('smooth', 0))
        if smooth > 1:
            curve_vals = _moving_average(curve_vals, smooth)
        lo, hi = float(spec['min']), float(spec['max'])
        values = lo + np.clip(curve_vals, 0.0, 1.0) * (hi - lo)
        if spec.get('invert'):
            values = lo + hi - values
        quantize = spec.get('quantize')
        if quantize:
            values = np.round(values / float(quantize)) * float(quantize)
        values = np.clip(values, min(lo, hi), max(lo, hi))
        # Punch last, after clipping: a punch value is an explicit override
        # and may deliberately sit outside the mapped range (the default
        # flow_blend punch is 0.0, below its 0.35 floor).
        punch = spec.get('onset_punch')
        if punch:
            values = _apply_onset_punch(values, onsets, punch)
        if name == 'steps_schedule':
            schedules[name] = [int(round(float(v))) for v in values]
        else:
            schedules[name] = [round(float(v), 3) for v in values]
    return schedules


def apply_audio_schedules(config, schedules: Dict[str, list]) -> None:
    """Apply compiled audio schedules to a run config, in place.

    Same contract as scene_detect.apply_content_schedules: sets
    config.diffusion.{steps,cfg_scale,style_strength}_schedule and
    config.warp.flow_blend_schedule for the keys present.
    """
    for name, values in schedules.items():
        section_name, attr = SCHEDULE_TARGETS[name]
        setattr(getattr(config, section_name), attr, list(values))
    print('Applied audio-reactive schedules:')
    for name, values in schedules.items():
        if len(values) > 2:
            print(f'  {name}: {values[:12]}... ({len(values)} frames)')


def analyze_audio(path: str, fps: float, num_frames: int,
                  sample_rate: int = DEFAULT_SAMPLE_RATE
                  ) -> Dict[str, List[float]]:
    """Decode an audio/video file and return its per-frame features."""
    samples, sr = decode_audio(path, sample_rate=sample_rate)
    return frame_features(samples, sr, fps, num_frames)
