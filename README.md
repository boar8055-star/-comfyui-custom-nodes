# ComfyUI-LTXTimedTracks

Custom nodes for retiming LTX sparse motion tracks.

## Nodes

### 1) LTX Track Range Retimer
Takes the `tracks` output from `LTXVSparseTrackEditor` and plays the first N fish tracks only inside a chosen frame range.

Good for:
- `start_frame = 48`, `end_frame = 70`
- keep water tracks untouched
- hold the fish still before/after the active range

### 2) LTX Track Keyframe Retimer
Takes the `tracks` output from `LTXVSparseTrackEditor` and retimes fish tracks by explicit `progress@frame` pairs.

Example schedule:

```text
1: 0@48, 0.33@56, 0.66@62, 1@70
2: 0@48, 0.33@56, 0.66@62, 1@70
3: 0@48, 0.33@56, 0.66@62, 1@70
4: 0@48, 0.33@56, 0.66@62, 1@70
```

This means: run the drawn path progress from 0 to 1 between the listed frames.

## Install

Copy this folder into:

```text
ComfyUI/custom_nodes/ComfyUI-LTXTimedTracks
```

Then restart ComfyUI.

## Suggested wiring

```text
LTXVSparseTrackEditor -> LTX Track Range Retimer -> LTX Generate (tracks)
```

or

```text
LTXVSparseTrackEditor -> LTX Track Keyframe Retimer -> LTX Generate (tracks)
```
