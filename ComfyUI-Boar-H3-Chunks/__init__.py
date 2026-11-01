"""Boar H3 native ComfyUI loop helpers. No web/API calls."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile

import torch
import folder_paths

FPS = 24
CHUNK_INPUT = 175       # first H3 loses 17 frames, second H3 loses 17
CHUNK_OUTPUT = 141      # 175-17-17 = 141 24fps frames, step of the source timeline
FINAL_FPS = 30


def run(*args):
    p = subprocess.run([str(x) for x in args], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"ffmpeg failed ({p.returncode}): {' '.join(map(str, args))}\n{p.stderr[-4500:]}")
    return p.stdout


def video_files():
    base = Path(folder_paths.get_input_directory())
    extensions = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'}
    names = sorted(str(f.relative_to(base)).replace('\\', '/') for f in base.rglob('*')
                   if f.is_file() and f.suffix.lower() in extensions)
    return names or ['PUT_VIDEO_IN_COMFYUI_INPUT.mp4']


def check_binary():
    for n in ('ffmpeg', 'ffprobe'):
        if shutil.which(n) is None:
            raise RuntimeError(f'{n} is required on PATH. Install ffmpeg on this ComfyUI host.')


def resolve_input(name):
    base = Path(folder_paths.get_input_directory()).resolve()
    candidate = (base / name).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        raise ValueError(f'Video must exist under {base}: {name!r}')
    return candidate


def ffprobe(path):
    return json.loads(run('ffprobe','-v','error','-show_streams','-show_format','-of','json',path))


def video_stream(path):
    data = ffprobe(path)
    stream = next((x for x in data['streams'] if x['codec_type'] == 'video'), None)
    if stream is None:
        raise ValueError(f'No video stream in {path}')
    return stream, data


class BoarVideoSource:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'video': (video_files(),)}}

    RETURN_TYPES = ('BOAR_VIDEO_SOURCE', 'INT', 'STRING')
    RETURN_NAMES = ('source', 'number_of_chunks', 'info')
    FUNCTION = 'prepare'
    CATEGORY = 'BOAR / H3 chunks'

    @classmethod
    def IS_CHANGED(cls, video):
        f = resolve_input(video)
        st = f.stat()
        return f'{st.st_size}:{st.st_mtime_ns}'

    def prepare(self, video):
        check_binary()
        original = resolve_input(video)
        st = original.stat()
        digest = hashlib.sha256(f'{original}:{st.st_mtime_ns}:{st.st_size}:fps24_v3'.encode()).hexdigest()[:20]
        work_dir = Path(folder_paths.get_temp_directory()) / 'boar_h3_chunks' / digest
        work_dir.mkdir(parents=True, exist_ok=True)
        normalized = work_dir / 'normalized_24fps.mp4'
        if not normalized.exists() or normalized.stat().st_size < 512:
            temp = work_dir / 'normalizing.tmp.mp4'
            if temp.exists():
                temp.unlink()
            run('ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', original,
                '-map', '0:v:0', '-map', '0:a:0?', '-vf', 'fps=24',
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '16', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
                '-movflags', '+faststart', temp)
            temp.replace(normalized)
        stream, meta = video_stream(normalized)
        frames = int(stream.get('nb_frames') or 0)
        if frames < 1:
            raise ValueError('Could not get normalized frame count from ffprobe.')
        duration = float(frames / FPS)
        source = {'normalized': str(normalized), 'original': str(original),
                  'work_dir': str(work_dir), 'frames': frames, 'duration': duration,
                  'has_audio': any(x['codec_type'] == 'audio' for x in meta['streams']),
                  'token': digest}
        chunks = math.ceil(frames / CHUNK_OUTPUT)
        return (source, chunks, f'{frames} frames at 24fps, {chunks} chunks, ~{duration:.3f}s')


class BoarLoadChunk:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'source': ('BOAR_VIDEO_SOURCE',),
            'chunk_index': ('INT', {'default': 0, 'min': 0, 'forceInput': True}),
            'max_image_side': ('INT', {'default': 1344, 'min': 512, 'max': 2048, 'step': 32}),
        }}

    RETURN_TYPES = ('IMAGE', 'AUDIO', 'FLOAT')
    RETURN_NAMES = ('frames_175', 'padded_audio', 'fps_24')
    FUNCTION = 'load'
    CATEGORY = 'BOAR / H3 chunks'

    def load(self, source, chunk_index, max_image_side):
        check_binary()
        index = int(chunk_index)
        if index < 0 or index * CHUNK_OUTPUT >= source['frames']:
            raise ValueError(f'Invalid chunk index {index}; original frames {source["frames"]}')
        path = source['normalized']
        v, _ = video_stream(path)
        w, h = int(v['width']), int(v['height'])
        factor = min(1., float(max_image_side) / max(w, h))
        w = max(32, (int(w * factor) // 2) * 2)
        h = max(32, (int(h * factor) // 2) * 2)
        duration = CHUNK_INPUT / FPS
        # Pre-normalization makes frame-accurate seeking possible at 24 CFR.
        start = index * CHUNK_OUTPUT / FPS
        cmd = ['ffmpeg','-hide_banner','-loglevel','error','-ss',f'{start:.9f}',
               '-i',path,'-an','-vf',f'scale={w}:{h}:flags=lanczos,tpad=stop_mode=clone:stop_duration=10',
               '-frames:v',str(CHUNK_INPUT),'-f','rawvideo','-pix_fmt','rgb24','pipe:1']
        p = subprocess.run(cmd, capture_output=True)
        if p.returncode:
            raise RuntimeError(p.stderr.decode(errors='replace')[-3500:])
        expected = CHUNK_INPUT * w * h * 3
        if len(p.stdout) != expected:
            raise RuntimeError(f'Video chunk {index}: expected {expected} RGB bytes, received {len(p.stdout)}')
        # torch.frombuffer() must use a writable backing store.
        image = torch.frombuffer(bytearray(p.stdout), dtype=torch.uint8).view(CHUNK_INPUT,h,w,3)
        images = image.float().div_(255.0)
        audio_samples = round(duration * 48000)
        if source['has_audio']:
            c = ['ffmpeg','-hide_banner','-loglevel','error','-ss',f'{start:.9f}',
                 '-i',path,'-vn','-af',f'apad,atrim=duration={duration:.9f}',
                 '-ar','48000','-ac','2','-f','f32le','pipe:1']
            ap = subprocess.run(c, capture_output=True)
            if ap.returncode:
                raise RuntimeError(ap.stderr.decode(errors='replace')[-3500:])
            audio = torch.frombuffer(bytearray(ap.stdout), dtype=torch.float32)
            audio = audio[:audio_samples*2]
            if audio.numel() < audio_samples*2:
                audio = torch.nn.functional.pad(audio, (0, audio_samples*2-audio.numel()))
            audio = audio.reshape(audio_samples,2).T.contiguous().unsqueeze(0)
        else:
            audio = torch.zeros((1,2,audio_samples),dtype=torch.float32)
        return (images, {'waveform': audio, 'sample_rate': 48000}, float(FPS))


class BoarAppendChunk:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'previous_chunks': ('BOAR_CHUNKS', {'forceInput': True}),
            'chunk_index': ('INT', {'forceInput': True}),
            'finished_mp4_path': ('STRING', {'forceInput': True}),
        }}

    RETURN_TYPES = ('BOAR_CHUNKS',)
    RETURN_NAMES = ('chunks_for_next_iteration',)
    FUNCTION = 'append'
    CATEGORY = 'BOAR / H3 chunks'

    def append(self, previous_chunks, chunk_index, finished_mp4_path):
        path = Path(finished_mp4_path)
        if not path.is_file() or path.stat().st_size < 512:
            raise RuntimeError(f'Final chunk video was not generated: {path}')
        # Always copy: loops must not mutate a previously cached state.
        old = [] if previous_chunks is None else list(previous_chunks)
        index = int(chunk_index)
        if any(int(x[0]) == index for x in old):
            raise RuntimeError(f'Duplicate chunk {index}: check Start/End Loop wiring.')
        return (old + [(index, str(path))],)


class BoarJoinChunks:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'source': ('BOAR_VIDEO_SOURCE',),
            'chunks': ('BOAR_CHUNKS', {'forceInput': True}),
            'output_filename': ('STRING', {'default':'H3_BOAR_FINAL_1080x1920_30fps.mp4'}),
            'crf': ('INT', {'default':16, 'min':0, 'max':35}),
        }}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('final_video_path',)
    FUNCTION = 'join'
    OUTPUT_NODE = True
    CATEGORY = 'BOAR / H3 chunks'

    def join(self, source, chunks, output_filename, crf):
        check_binary()
        items = sorted(list(chunks or []), key=lambda x: x[0])
        n = math.ceil(source['frames']/CHUNK_OUTPUT)
        if [int(x[0]) for x in items] != list(range(n)):
            raise RuntimeError(f'Expected chunk indices 0..{n-1}, got {[x[0] for x in items]}')
        session = Path(source['work_dir'])
        joined_dir = session/'joined'
        joined_dir.mkdir(parents=True, exist_ok=True)
        clean = []
        # Round cumulative time, not each chunk independently: no sync drift.
        for i, filename in items:
            start24 = int(i) * CHUNK_OUTPUT
            end24 = min(start24+CHUNK_OUTPUT, int(source['frames']))
            # round(x) is banker's; use half-up for deterministic video frame positions
            start30 = (start24*FINAL_FPS*2 + FPS)//(2*FPS)
            end30 = (end24*FINAL_FPS*2 + FPS)//(2*FPS)
            frames30 = end30 - start30
            if frames30 <= 0:
                raise RuntimeError('Zero-length chunk encountered.')
            target = joined_dir / f'clip_{i:06d}.mp4'
            # Last-framed pad guards against slightly different RIFE interpolation counts.
            run('ffmpeg','-hide_banner','-loglevel','error','-y','-i',filename,
                '-an','-vf',f'fps=30,tpad=stop_mode=clone:stop_duration=1,trim=end_frame={frames30},setpts=PTS-STARTPTS',
                '-frames:v',str(frames30),'-c:v','libx264','-preset','veryfast',
                '-crf',str(crf),'-pix_fmt','yuv420p','-r','30',target)
            actual_stream,_=video_stream(target)
            if int(actual_stream.get('nb_frames') or 0)!=frames30:
                raise RuntimeError(f'Joined clip {i}: expected {frames30} frames.')
            clean.append(target)
        concat_file = joined_dir/'concat.txt'
        def quote(p):
            return "file '"+str(p).replace("'", "'\\''")+"'"
        concat_file.write_text('\n'.join(quote(p.resolve()) for p in clean)+'\n', encoding='utf-8')
        stem = Path(output_filename).name
        if not stem.lower().endswith('.mp4'):
            stem += '.mp4'
        dest = Path(folder_paths.get_output_directory()) / stem
        dest.parent.mkdir(parents=True,exist_ok=True)
        temp = joined_dir/'video_only.mp4'
        run('ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0',
            '-i',concat_file,'-an','-c:v','copy',temp)
        original = source['original']
        original_has_audio = any(x['codec_type']=='audio' for x in ffprobe(original)['streams'])
        if original_has_audio:
            run('ffmpeg','-hide_banner','-loglevel','error','-y','-i',temp,'-i',original,
                '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k',
                '-t',f'{source["frames"]/FPS:.9f}','-movflags','+faststart',dest)
        else:
            run('ffmpeg','-hide_banner','-loglevel','error','-y','-i',temp,
                '-c:v','copy','-an','-movflags','+faststart',dest)
        stream,_ = video_stream(dest)
        actual=int(stream.get('nb_frames') or 0)
        expected=(source['frames']*FINAL_FPS*2+FPS)//(2*FPS)
        if actual != expected:
            raise RuntimeError(f'Final output length mismatch: {actual} != {expected} frames')
        print(f'[BOAR/H3] FINISHED: {dest} ({actual} frames at 30fps)')
        return (str(dest),)


NODE_CLASS_MAPPINGS = {
    'BoarH3VideoSource':BoarVideoSource,
    'BoarH3LoadChunk':BoarLoadChunk,
    'BoarH3AppendChunk':BoarAppendChunk,
    'BoarH3JoinChunks':BoarJoinChunks,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'BoarH3VideoSource':'BOAR H3 • 1 Select video / normalize 24fps',
    'BoarH3LoadChunk':'BOAR H3 • 2 Load 175-frame window (<=7s generated)',
    'BoarH3AppendChunk':'BOAR H3 • 3 Collect finished segment path',
    'BoarH3JoinChunks':'BOAR H3 • 4 Join videos / original audio',
}
