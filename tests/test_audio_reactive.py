"""CPU tests for audio-reactive scheduling (core/audio_reactive.py).

All signals are synthetic numpy arrays; no GPU, no network. The decode test
writes a real .wav with the stdlib wave module.
"""

import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from vibewarp.config import RunConfig
from vibewarp.core.audio_reactive import (
    DEFAULT_MAPPING, analyze_audio, apply_audio_schedules, compile_schedules,
    decode_audio, detect_onsets, estimate_tempo, frame_features)

SR = 22050
FPS = 24.0


def _burst_track(burst_frames, num_frames, sr=SR, fps=FPS):
    """Silence with a loud broadband burst at each given frame."""
    hop = int(round(sr / fps))
    samples = np.zeros(hop * num_frames, dtype=np.float32)
    rng = np.random.default_rng(7)
    for frame in burst_frames:
        start = frame * hop
        samples[start:start + hop // 2] = rng.normal(0, 0.9, hop // 2)
    return samples


def test_onsets_found_at_bursts():
    bursts = [10, 40, 70]
    samples = _burst_track(bursts, 96)
    features = frame_features(samples, SR, FPS, 96)
    onsets = detect_onsets(features['onset'])
    assert onsets, 'expected onsets in a burst track'
    for burst in bursts:
        assert any(abs(o - burst) <= 1 for o in onsets), \
            f'no onset within 1 frame of burst at {burst}: {onsets}'


def test_steady_sine_schedules_within_bounds():
    t = np.arange(SR * 4, dtype=np.float64) / SR
    samples = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    features = frame_features(samples, SR, FPS, 96)
    schedules = compile_schedules(features, FPS)
    assert set(schedules) == set(DEFAULT_MAPPING)
    for name, spec in DEFAULT_MAPPING.items():
        values = schedules[name]
        assert len(values) == 96
        assert max(values) <= max(spec['min'], spec['max']) + 1e-9
        if name == 'flow_blend_schedule':
            # The onset punch deliberately sits below the mapped floor.
            assert min(values) >= 0.0
        else:
            assert min(values) >= min(spec['min'], spec['max']) - 1e-9
    assert all(isinstance(v, int) for v in schedules['steps_schedule'])


def test_flow_blend_onset_punch_and_recovery():
    bursts = [20]
    samples = _burst_track(bursts, 96)
    features = frame_features(samples, SR, FPS, 96)
    onsets = detect_onsets(features['onset'])
    assert onsets
    schedules = compile_schedules(features, FPS)
    flow = schedules['flow_blend_schedule']
    first_onset = onsets[0]
    assert flow[first_onset] == pytest.approx(0.0, abs=0.05)
    # Four frames after the punch the value has decayed back to the curve,
    # so it must be clearly above the punched value.
    assert flow[first_onset + 4] > flow[first_onset]


def test_estimate_tempo_periodic_onsets():
    # Peaks every 12 frames at 24 fps = one beat every 0.5 s = 120 BPM.
    curve = [0.0] * 240
    for i in range(0, 240, 12):
        curve[i] = 1.0
    tempo = estimate_tempo(curve, FPS)
    assert tempo is not None
    assert abs(tempo - 120.0) <= 6.0


def test_estimate_tempo_silence_is_none():
    assert estimate_tempo([0.0] * 100, FPS) is None


def test_apply_audio_schedules_sets_config():
    features = frame_features(_burst_track([5], 48), SR, FPS, 48)
    schedules = compile_schedules(features, FPS)
    config = RunConfig()
    apply_audio_schedules(config, schedules)
    assert config.diffusion.steps_schedule == schedules['steps_schedule']
    assert config.diffusion.cfg_scale_schedule == schedules['cfg_scale_schedule']
    assert config.diffusion.style_strength_schedule == \
        schedules['style_strength_schedule']
    assert config.warp.flow_blend_schedule == schedules['flow_blend_schedule']


def test_short_audio_pads_to_num_frames():
    samples = _burst_track([2], 10)
    features = frame_features(samples, SR, FPS, 40)
    assert all(len(v) == 40 for v in features.values())
    assert features['onset'][20] == 0.0  # beyond the audio: no phantom onsets


def test_decode_audio_wav_roundtrip(tmp_path):
    path = tmp_path / 'tone.wav'
    t = np.arange(SR, dtype=np.float64) / SR
    pcm = (0.4 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    samples, sr = decode_audio(str(path))
    assert sr == SR
    assert abs(samples.size - SR) <= SR // 10
    assert np.abs(samples).max() > 0.2


def test_analyze_audio_end_to_end(tmp_path):
    path = tmp_path / 'bursts.wav'
    samples = _burst_track([4, 12], 24)
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    features = analyze_audio(str(path), FPS, 24)
    assert all(len(v) == 24 for v in features.values())


def test_cli_writes_schedules_json(tmp_path):
    wav_path = tmp_path / 'bursts.wav'
    samples = _burst_track([8, 24, 40, 56, 72, 88], 96)
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(wav_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    out_path = tmp_path / 'schedules.json'
    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, str(repo_root / 'tools' / 'audio_schedule.py'),
         '--audio', str(wav_path), '--fps', '24', '--frames', '96',
         '--out', str(out_path)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_path.read_text())
    for name in DEFAULT_MAPPING:
        assert len(payload[name]) == 96
    assert payload['meta']['num_frames'] == 96
    assert payload['meta']['onsets']
