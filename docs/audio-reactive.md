# Audio-reactive scheduling

Content-aware scheduling watches the *picture* (scene changes) and rewrites
schedules at cuts. Audio-reactive scheduling watches the *music* and rewrites
the same schedules from the sound. A render cut to music can now breathe with
it: quiet passages hold a gentler style, loud passages push harder, and a beat
hit can drop warp inheritance for one frame so the image lands fresh on the
hit, then fades back.

Code: `vibewarp/core/audio_reactive.py`. Standalone compiler:
`tools/audio_schedule.py`.

## How it works

1. The audio (a separate file, or the source video's own audio track) is
   decoded to mono samples with ffmpeg. If ffmpeg is missing, .wav files fall
   back to the Python standard library. No new dependencies are introduced.
2. The audio is cut into one window per video frame and three features are
   measured per frame, each robustly normalized to 0 to 1:
   - **energy**: RMS loudness of the window.
   - **bass**: share of the spectrum below 250 Hz.
   - **onset**: spectral flux, how much the spectrum just changed. Peaks in
     this curve are detected as onsets (beat hits, attacks, cuts in the music).
3. A *mapping* compiles each feature into a schedule: a per-frame list for one
   of the four schedulable parameters (`steps`, `cfg_scale`,
   `style_strength`, `flow_blend`). A mapping spec sets the value range, an
   optional curve (linear, pow2, sqrt), smoothing in frames, inversion, an
   optional quantize step, and an optional **onset punch**: at each detected
   onset the value jumps to a set point and decays linearly back to the mapped
   curve over a few frames, the same falloff idea scene scheduling uses at
   scene cuts.

## The default preset (Lightbrush)

| Schedule | Source | Range | Extras |
|---|---|---|---|
| style_strength | energy | 0.45 to 0.85 | smooth 3 frames |
| cfg_scale | energy | 5.0 to 9.0 | smooth 3 frames |
| flow_blend | bass | 0.35 to 0.80 | smooth 2, onset punch to 0.0 decaying over 4 frames |
| steps | onset | 15 to 30 | quantized to whole steps |

Reading it musically: loud means more stylized and more prompt-faithful;
heavy bass means the warped previous frame carries further; a hit means one
fresh frame, then the warp fades back in; busy transients get more steps.

## Use it in a render

Set the `audio` section of the run config:

- `enabled`: true turns it on. Default is false; nothing changes when off.
- `audio_path`: an audio file, or a video file to take the audio from.
  Empty means the source video's own audio track.
- `fps_override`: 0 uses the probed output fps.
- `mappings`: a custom mappings dict, or null for the default preset.
- `onset_threshold_percentile`: onset sensitivity, default 90.

In flat VibeWarp settings files these are `audio_enabled`, `audio_audio_path`,
`audio_fps_override`, `audio_mappings`, and
`audio_onset_threshold_percentile`.

Audio problems never kill a render: the pipeline prints a warning and
continues without audio schedules.

## Use it standalone

Compile schedules without rendering, to inspect them, reuse them, or paste
them into a settings file:

```bash
python tools/audio_schedule.py --audio song.mp3 --fps 24 --frames 240 --out schedules.json
```

The JSON holds the four schedules plus a `meta` block with the detected onset
frames and a tempo estimate. A custom mapping can be supplied with
`--mappings my_mapping.json` (a JSON object in the same shape as the preset
table above).

## Limitations, honestly

- Tempo is a simple autocorrelation estimate over the onset curve. It is a
  useful readout, not a beat tracker, and rubato or heavily syncopated music
  will fool it.
- Features are per-frame windows, so timing resolution is one video frame.
- Bass is a single 250 Hz split, not a full band decomposition.
- There is deliberately no librosa or learned model here: the module is
  numpy and ffmpeg only, so it runs anywhere VibeWarp runs, including CPU
  only machines, and adds nothing to the install.
