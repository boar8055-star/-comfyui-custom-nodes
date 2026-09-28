import json
from typing import List, Dict, Tuple


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _parse_tracks(tracks_json: str) -> List[List[Dict[str, int]]]:
    if isinstance(tracks_json, list):
        return tracks_json
    return json.loads(tracks_json)


def _to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _sample_track(track: List[Dict[str, int]], progress: float) -> Dict[str, int]:
    """Sample a pre-sampled track by normalized progress [0,1]."""
    if not track:
        return {"x": 0, "y": 0}
    if len(track) == 1:
        return {"x": int(round(track[0]["x"])), "y": int(round(track[0]["y"]))}
    p = _clamp(progress, 0.0, 1.0) * (len(track) - 1)
    i0 = int(p)
    i1 = min(i0 + 1, len(track) - 1)
    t = p - i0
    x = track[i0]["x"] * (1 - t) + track[i1]["x"] * t
    y = track[i0]["y"] * (1 - t) + track[i1]["y"] * t
    return {"x": int(round(x)), "y": int(round(y))}


def _scale_from_start(track: List[Dict[str, int]], sample: Dict[str, int], amp: float) -> Dict[str, int]:
    if not track:
        return sample
    x0, y0 = track[0]["x"], track[0]["y"]
    dx = sample["x"] - x0
    dy = sample["y"] - y0
    return {"x": int(round(x0 + dx * amp)), "y": int(round(y0 + dy * amp))}


class LTXTrackRangeRetimer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracks": ("STRING", {"forceInput": True}),
                "total_frames": ("INT", {"default": 361, "min": 2, "max": 10000}),
                "fish_track_count": ("INT", {"default": 4, "min": 1, "max": 64}),
                "start_frame": ("INT", {"default": 48, "min": 0, "max": 99999}),
                "end_frame": ("INT", {"default": 96, "min": 1, "max": 99999}),
                "head_amp": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 5.0, "step": 0.01}),
                "front_amp": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 5.0, "step": 0.01}),
                "rear_amp": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 5.0, "step": 0.01}),
                "tail_amp": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 5.0, "step": 0.01}),
                "after_mode": (["hold_last", "return_to_start"], {"default": "hold_last"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tracks",)
    FUNCTION = "retime"
    CATEGORY = "LTX/Tracks"

    def retime(self, tracks, total_frames, fish_track_count, start_frame, end_frame,
               head_amp, front_amp, rear_amp, tail_amp, after_mode):
        parsed = _parse_tracks(tracks)
        start = max(0, min(start_frame, total_frames - 1))
        end = max(start + 1, min(end_frame, total_frames - 1))
        amps = [head_amp, front_amp, rear_amp, tail_amp]
        out = []
        for ti, tr in enumerate(parsed):
            if ti >= fish_track_count:
                # pass through water/other tracks unchanged
                out.append(tr)
                continue
            amp = amps[ti] if ti < len(amps) else amps[-1]
            new_tr = []
            start_pt = _scale_from_start(tr, _sample_track(tr, 0.0), amp)
            end_pt = _scale_from_start(tr, _sample_track(tr, 1.0), amp)
            for f in range(total_frames):
                if f < start:
                    new_tr.append(start_pt)
                elif f <= end:
                    prog = (f - start) / max(1, (end - start))
                    pt = _scale_from_start(tr, _sample_track(tr, prog), amp)
                    new_tr.append(pt)
                else:
                    if after_mode == "hold_last":
                        new_tr.append(end_pt)
                    else:
                        # linear settle back to start by final frame
                        back_t = (f - end) / max(1, total_frames - 1 - end)
                        x = end_pt["x"] * (1 - back_t) + start_pt["x"] * back_t
                        y = end_pt["y"] * (1 - back_t) + start_pt["y"] * back_t
                        new_tr.append({"x": int(round(x)), "y": int(round(y))})
            out.append(new_tr)
        return (_to_json(out),)


class LTXTrackKeyframeRetimer:
    """
    Retimes tracks using explicit progress@frame pairs.
    Syntax example:
      1: 0@48, 0.33@56, 0.66@62, 1@70
      2: 0@48, 0.33@56, 0.66@62, 1@70
    Track numbering starts at 1.
    Progress is normalized along the drawn path (0=start, 1=end).
    """

    DEFAULT_SCHEDULE = (
        "# One line per fish track.\n"
        "# Format: track_number: progress@frame, progress@frame, ...\n"
        "# Example: make a quick zig-zag between frames 48 and 70\n"
        "1: 0@48, 0.33@56, 0.66@62, 1@70\n"
        "2: 0@48, 0.33@56, 0.66@62, 1@70\n"
        "3: 0@48, 0.33@56, 0.66@62, 1@70\n"
        "4: 0@48, 0.33@56, 0.66@62, 1@70"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracks": ("STRING", {"forceInput": True}),
                "total_frames": ("INT", {"default": 361, "min": 2, "max": 10000}),
                "fish_track_count": ("INT", {"default": 4, "min": 1, "max": 64}),
                "schedule": ("STRING", {"multiline": True, "default": cls.DEFAULT_SCHEDULE}),
                "head_amp": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 5.0, "step": 0.01}),
                "front_amp": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 5.0, "step": 0.01}),
                "rear_amp": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 5.0, "step": 0.01}),
                "tail_amp": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 5.0, "step": 0.01}),
                "after_mode": (["hold_last", "return_to_start"], {"default": "hold_last"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tracks",)
    FUNCTION = "retime"
    CATEGORY = "LTX/Tracks"

    def _parse_schedule(self, schedule: str) -> Dict[int, List[Tuple[float, int]]]:
        result: Dict[int, List[Tuple[float, int]]] = {}
        for raw in schedule.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            try:
                track_idx = int(k.strip())
            except ValueError:
                continue
            pairs = []
            for token in v.split(","):
                token = token.strip()
                if not token or "@" not in token:
                    continue
                prog_s, frame_s = token.split("@", 1)
                try:
                    prog = float(prog_s.strip())
                    frame = int(frame_s.strip())
                    pairs.append((_clamp(prog, 0.0, 1.0), frame))
                except ValueError:
                    continue
            pairs.sort(key=lambda x: x[1])
            if pairs:
                result[track_idx] = pairs
        return result

    def retime(self, tracks, total_frames, fish_track_count, schedule,
               head_amp, front_amp, rear_amp, tail_amp, after_mode):
        parsed = _parse_tracks(tracks)
        sched = self._parse_schedule(schedule)
        amps = [head_amp, front_amp, rear_amp, tail_amp]
        out = []
        for ti, tr in enumerate(parsed):
            track_num = ti + 1
            if ti >= fish_track_count or track_num not in sched:
                out.append(tr)
                continue
            amp = amps[ti] if ti < len(amps) else amps[-1]
            pairs = sched[track_num]
            new_tr = []
            start_hold = _scale_from_start(tr, _sample_track(tr, 0.0), amp)
            last_pt = start_hold
            # fill whole clip by walking schedule segments
            first_frame = max(0, min(total_frames - 1, pairs[0][1]))
            for f in range(0, first_frame):
                new_tr.append(start_hold)
            # schedule segments
            for seg_i in range(len(pairs) - 1):
                p0, f0 = pairs[seg_i]
                p1, f1 = pairs[seg_i + 1]
                f0 = max(0, min(total_frames - 1, f0))
                f1 = max(0, min(total_frames - 1, f1))
                if len(new_tr) < f0:
                    pad_pt = _scale_from_start(tr, _sample_track(tr, p0), amp)
                    while len(new_tr) < f0:
                        new_tr.append(pad_pt)
                for f in range(f0, min(f1, total_frames - 1) + 1):
                    t = 0.0 if f1 == f0 else (f - f0) / (f1 - f0)
                    prog = p0 * (1 - t) + p1 * t
                    pt = _scale_from_start(tr, _sample_track(tr, prog), amp)
                    if len(new_tr) <= f:
                        new_tr.append(pt)
                    else:
                        new_tr[f] = pt
                    last_pt = pt
            # fill rest
            while len(new_tr) < total_frames:
                if after_mode == "hold_last":
                    new_tr.append(last_pt)
                else:
                    back_t = (len(new_tr) - (pairs[-1][1])) / max(1, total_frames - 1 - pairs[-1][1])
                    x = last_pt["x"] * (1 - back_t) + start_hold["x"] * back_t
                    y = last_pt["y"] * (1 - back_t) + start_hold["y"] * back_t
                    new_tr.append({"x": int(round(x)), "y": int(round(y))})
            out.append(new_tr[:total_frames])
        return (_to_json(out),)


NODE_CLASS_MAPPINGS = {
    "LTXTrackRangeRetimer": LTXTrackRangeRetimer,
    "LTXTrackKeyframeRetimer": LTXTrackKeyframeRetimer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXTrackRangeRetimer": "LTX Track Range Retimer",
    "LTXTrackKeyframeRetimer": "LTX Track Keyframe Retimer",
}
