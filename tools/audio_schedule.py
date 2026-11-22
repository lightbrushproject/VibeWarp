#!/usr/bin/env python3
"""Compile audio-reactive schedules without running a render.

Usage:
    python tools/audio_schedule.py --audio song.mp3 --fps 24 --frames 240 \
        --out schedules.json

Writes a JSON object with the compiled per-frame schedules (keys are the
RunConfig schedule field names) plus a "meta" block with the detected tempo,
onset frames, and the analysis parameters. The schedules can be applied to a
run with vibewarp.core.audio_reactive.apply_audio_schedules, or pasted into
a settings file as list schedules.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vibewarp.core.audio_reactive import (  # noqa: E402
    DEFAULT_MAPPING, analyze_audio, compile_schedules, detect_onsets,
    estimate_tempo)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Compile audio-reactive VibeWarp schedules from an '
                    'audio (or video) file.')
    parser.add_argument('--audio', required=True,
                        help='Audio file, or a video file with an audio track')
    parser.add_argument('--fps', required=True, type=float,
                        help='Video frame rate the schedules are compiled for')
    parser.add_argument('--frames', required=True, type=int,
                        help='Number of video frames to schedule')
    parser.add_argument('--out', required=True, help='Output JSON path')
    parser.add_argument('--preset', default='default',
                        choices=['default'],
                        help='Mapping preset (only "default" for now)')
    parser.add_argument('--mappings', default=None,
                        help='Optional JSON file with a custom mappings dict, '
                             'overriding the preset')
    parser.add_argument('--onset-threshold-percentile', type=float,
                        default=90.0)
    args = parser.parse_args(argv)

    mappings = DEFAULT_MAPPING
    if args.mappings:
        with open(args.mappings, 'r', encoding='utf-8') as fh:
            mappings = json.load(fh)

    features = analyze_audio(args.audio, args.fps, args.frames)
    schedules = compile_schedules(
        features, args.fps, mappings=mappings,
        onset_threshold_percentile=args.onset_threshold_percentile)
    onsets = detect_onsets(
        features['onset'],
        threshold_percentile=args.onset_threshold_percentile)
    tempo = estimate_tempo(features['onset'], args.fps)

    payload = dict(schedules)
    payload['meta'] = {
        'audio': os.path.abspath(args.audio),
        'fps': args.fps,
        'num_frames': args.frames,
        'tempo_bpm': tempo,
        'onsets': onsets,
    }
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=1)
    tempo_text = f'{tempo} BPM' if tempo else 'no clear tempo'
    print(f'Wrote {args.out}: {len(schedules)} schedules x {args.frames} '
          f'frames, {len(onsets)} onsets, {tempo_text}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
