"""Cycle through two independent media folders once per ComfyUI prompt.

Files are selected in natural-name order. Counts are persisted between runs.
"""

import json
import os
import re
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths

try:
    import fcntl  # POSIX flock: shared by multiple ComfyUI processes on the same machine
except ImportError:
    fcntl = None

_LOCAL_LOCK = threading.Lock()
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def _natural_key(path: Path):
    return [int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", path.name)]


def _resolve_folder(folder_text: str) -> Path:
    folder_text = os.path.expandvars(os.path.expanduser(folder_text.strip()))
    if not folder_text:
        raise ValueError("Folder name cannot be empty.")
    root = Path(folder_paths.get_input_directory())
    path = Path(folder_text)
    return (path if path.is_absolute() else root / path).resolve()


def _files(folder: Path, extensions: set[str], label: str) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(
            f"{label} directory does not exist: {folder}. "
            "Create it and put media files inside."
        )
    files = sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.casefold() in extensions),
        key=_natural_key,
    )
    if not files:
        raise ValueError(
            f"No supported {label} media files in {folder}. "
            f"Supported formats: {', '.join(sorted(extensions))}"
        )
    return files


def _increment_and_get(cycle_key: str) -> int:
    # Persist progress across queued generations, restarts, and multiple workers.
    # H3_FOLDER_CYCLE_STATE_DIR may point to a shared location for multiple installs.
    state_root = Path(os.environ.get("H3_FOLDER_CYCLE_STATE_DIR", folder_paths.get_input_directory()))
    state_root.mkdir(parents=True, exist_ok=True)
    state_path = state_root / ".h3_folder_cycle_counters.json"
    lock_path = state_root / ".h3_folder_cycle_counters.lock"

    with _LOCAL_LOCK:
        with lock_path.open("a+b") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                if state_path.exists():
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    if not isinstance(state, dict):
                        raise ValueError(f"Invalid counter file: {state_path}")
                else:
                    state = {}
                previous_count = int(state.get(cycle_key, 0))
                state[cycle_key] = previous_count + 1

                tmp_path = state_root / f".{state_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                try:
                    with tmp_path.open("w", encoding="utf-8") as out:
                        json.dump(state, out, ensure_ascii=False, indent=2)
                        out.flush()
                        os.fsync(out.fileno())
                    os.replace(tmp_path, state_path)
                finally:
                    tmp_path.unlink(missing_ok=True)
                return previous_count
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class H3FolderCycle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "body_folder": ("STRING", {"default": "auto_bodies", "multiline": False}),
                "video_folder": ("STRING", {"default": "auto_videos", "multiline": False}),
                "cycle_id": ("STRING", {"default": "main", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("body_image", "video_path", "selection_info", "generation_number")
    FUNCTION = "next_pair"
    CATEGORY = "MiniMax H3 / Batch"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Guarantee execution for every queued prompt, even with an unchanged seed.
        return float("nan")

    def next_pair(self, body_folder: str, video_folder: str, cycle_id: str):
        body_path = _resolve_folder(body_folder)
        video_path = _resolve_folder(video_folder)
        bodies = _files(body_path, IMAGE_EXT, "body")
        videos = _files(video_path, VIDEO_EXT, "video")

        # ONE count is advanced per generation, not once per output. Each folder
        # wraps around separately, so folder sizes can be different.
        key = json.dumps([str(body_path), str(video_path), cycle_id], ensure_ascii=False)
        number_zero_based = _increment_and_get(key)
        selected_body = bodies[number_zero_based % len(bodies)]
        selected_video = videos[number_zero_based % len(videos)]

        with Image.open(selected_body) as original:
            img = ImageOps.exif_transpose(original)
            # Preserve visible appearance of transparent images on white background.
            if img.mode in ("RGBA", "LA") or "transparency" in img.info:
                img = img.convert("RGBA")
                canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(canvas, img).convert("RGB")
            else:
                img = img.convert("RGB")
            array = np.asarray(img, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)

        info = (
            f"GEN {number_zero_based + 1:04d} | "
            f"BODY {selected_body.name} ({(number_zero_based % len(bodies)) + 1}/{len(bodies)}) | "
            f"VIDEO {selected_video.name} ({(number_zero_based % len(videos)) + 1}/{len(videos)})"
        )
        print(f"[H3 Folder Cycle] {info}")
        return (tensor, str(selected_video), info, number_zero_based + 1)
