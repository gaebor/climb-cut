"""A small, deliberately dependency-light editor for multi-attempt climb videos.

The JSON descriptor is the source of truth.  ``preview`` is intentionally a
review surface, not a second project format: every saved adjustment remains a
short, reviewable JSON change.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    migrate_legacy_anchors(data)
    validate(data)
    return data


def save(path: Path, data: dict[str, Any]) -> None:
    order_anchors(data)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def order_anchors(d: dict[str, Any]) -> None:
    """Keep JSON anchor objects in shared output-time order for clean diffs."""
    hold_ids = [hold["id"] for hold in d.get("holds", [])]
    for track in d.get("tracks", []):
        anchors = track.get("anchors", {})
        ordered = {hold_id: anchors[hold_id] for hold_id in hold_ids if hold_id in anchors}
        # Preserve unexpected/legacy keys after known shared holds rather than
        # silently discarding user data.
        ordered.update({key: value for key, value in anchors.items() if key not in ordered})
        track["anchors"] = ordered


def validate(d: dict[str, Any]) -> None:
    if d.get("version") != 1 or not d.get("tracks") or not d.get("holds"):
        raise ValueError("descriptor needs version: 1, holds, and at least one track")
    holds = d["holds"]
    if [x["at"] for x in holds] != sorted(x["at"] for x in holds):
        raise ValueError("holds must be ordered by their output 'at' time")
    ids = [x["id"] for x in holds]
    if len(ids) != len(set(ids)):
        raise ValueError("hold ids must be unique")
    if float(d.get("output", {}).get("fps", 30)) != 30:
        raise ValueError("output.fps must be 30; climb-cut emits a fixed 30 FPS timeline")
    anchor_ids = [x["id"] for x in alignment_holds(d)]
    if not anchor_ids:
        raise ValueError("descriptor needs at least one alignment hold")
    if output_duration(d) < d["holds"][-1]["at"]:
        raise ValueError("output.duration cannot end before the last listed hold")
    for tr in d["tracks"]:
        if not tr.get("source") or not tr.get("anchors"):
            raise ValueError(f"track {tr.get('id', '?')} needs source and anchors")
        missing = set(anchor_ids) - set(tr["anchors"])
        if missing:
            raise ValueError(f"track {tr.get('id', '?')} lacks anchors: {sorted(missing)}")
        for hold_id in anchor_ids:
            anchor = tr["anchors"][hold_id]
            if not isinstance(anchor, dict) or not all(isinstance(anchor.get(key), (int, float)) for key in ("source", "opacity")):
                raise ValueError(f"track {tr.get('id', '?')} anchor {hold_id} needs numeric source and opacity")
        translation = tr.get("translation", [0, 0])
        if not isinstance(translation, list) or len(translation) != 2 or not all(isinstance(x, (int, float)) for x in translation):
            raise ValueError(f"track {tr.get('id', '?')} translation must be [x, y] pixels")
        if not isinstance(tr.get("enabled", True), bool):
            raise ValueError(f"track {tr.get('id', '?')} enabled must be true or false")


def lerp(points: list[list[float]], t: float) -> float:
    """Linear function, clamped to its end values."""
    points = sorted(points)
    if t <= points[0][0]: return float(points[0][1])
    if t >= points[-1][0]: return float(points[-1][1])
    for (a, av), (b, bv) in zip(points, points[1:]):
        if a <= t <= b:
            return av + (bv - av) * (t - a) / (b - a)
    return 0.0


def source_time(d: dict[str, Any], tr: dict[str, Any], output_t: float) -> float:
    """Map output time to source time; outside anchors playback is 1x."""
    pts = [(h["at"], anchor_source(tr, h["id"])) for h in alignment_holds(d)]
    if output_t <= pts[0][0]:
        return pts[0][1] + output_t - pts[0][0]
    if output_t >= pts[-1][0]:
        return pts[-1][1] + output_t - pts[-1][0]
    return lerp([[float(a), float(b)] for a, b in pts], output_t)


def alignment_holds(d: dict[str, Any]) -> list[dict[str, Any]]:
    """Legacy final `finish` is a duration marker, not a sync anchor."""
    holds = d["holds"]
    return holds[:-1] if len(holds) > 1 and holds[-1]["id"] == "finish" else holds


def output_duration(d: dict[str, Any]) -> float:
    """Explicit duration wins; legacy descriptors use their finish marker."""
    return float(d.get("output", {}).get("duration", d["holds"][-1]["at"]))


def anchor_source(track: dict[str, Any], hold_id: str) -> float:
    return float(track["anchors"][hold_id]["source"])


def anchor_opacity(track: dict[str, Any], hold_id: str) -> float:
    return float(track["anchors"][hold_id]["opacity"])


def set_anchor_source(d: dict[str, Any], track: dict[str, Any], hold: dict[str, Any], value: float) -> None:
    """Set a source timestamp; decreasing adjacent anchors create reverse playback."""
    track["anchors"][hold["id"]]["source"] = value


def raw_opacity(d: dict[str, Any], track: dict[str, Any], t: float, source_end: float | None = None) -> float:
    points = [[hold["at"], anchor_opacity(track, hold["id"])] for hold in alignment_holds(d)]
    if t <= points[-1][0] or source_end is None:
        return max(0.0, lerp(points, t))
    # Beyond the last synchronisation point, source time is already 1x.
    # Fade the last anchor's contribution to zero exactly when that source
    # clip reaches its final decoded presentation timestamp.
    last_output, last_value = points[-1]
    remaining = source_end - anchor_source(track, alignment_holds(d)[-1]["id"])
    if remaining <= 0: return 0.0
    return max(0.0, last_value * (1 - (t - last_output) / remaining))


def migrate_legacy_anchors(d: dict[str, Any]) -> None:
    """Read old numeric anchors/independent fade knots, but save the simpler form."""
    for track in d.get("tracks", []):
        anchors = track.get("anchors", {})
        if not any(isinstance(value, (int, float)) for value in anchors.values()):
            continue
        old_opacity = track.get("opacity", [[0, 1]])
        for hold in d.get("holds", []):
            hold_id = hold["id"]
            if isinstance(anchors.get(hold_id), (int, float)):
                anchors[hold_id] = {"source": anchors[hold_id], "opacity": max(0.0, lerp(old_opacity, hold["at"]))}
        track.pop("opacity", None)


def weights(d: dict[str, Any], t: float, source_ends: list[float] | None = None) -> list[float]:
    source_ends = source_ends or [None] * len(d["tracks"])
    raw = [raw_opacity(d, track, t, source_end) if track.get("enabled", True) else 0.0 for track, source_end in zip(d["tracks"], source_ends)]
    total = sum(raw)
    return [x / total for x in raw] if total else [0.0] * len(raw)


def metadata(video: Path) -> tuple[float, float, int, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise ValueError(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    result = (frames / fps, fps, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return result


def cache_path(descriptor_dir: Path, source: Path) -> Path:
    # Deliberately visible: users can inspect a clip's extracted timeline.
    # The short suffix keeps folders distinct if similarly named files occur.
    digest = hashlib.sha1(str(source.resolve()).encode()).hexdigest()[:8]
    return descriptor_dir / "clips" / f"{source.stem}-{digest}"


def source_signature(source: Path) -> dict[str, int]:
    stat = source.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def reported_frame_count(source: Path) -> int | None:
    """Best-effort container count for a cache progress bar."""
    cap = cv2.VideoCapture(str(source))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    finally:
        cap.release()


def build_frame_cache(source: Path, directory: Path, max_height: int = 720,
                      progress_queue: Any | None = None, progress_id: int | None = None) -> dict[str, Any]:
    """Decode once and save a timestamp-indexed, preview-sized JPEG cache."""
    directory.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened(): raise ValueError(f"Cannot open {source}")
    timestamps: list[float] = []
    files: list[str] = []
    frame_number = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    # A process may not write a tqdm bar directly: Windows terminals cannot
    # reliably interleave their cursor-control sequences.  In that case the
    # parent process owns the bars and receives small batched increments.
    progress = None if progress_queue is not None else tqdm(
        total=total, desc=f"Cache {source.name}", unit="frame", dynamic_ncols=True)
    pending_progress = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok: break
            # POS_MSEC is the decoder's presentation timestamp of this frame;
            # it is deliberately not derived from nominal FPS.
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            timestamps.append(timestamp)
            h, w = image.shape[:2]
            if h > max_height:
                image = cv2.resize(image, (round(w * max_height / h), max_height), interpolation=cv2.INTER_AREA)
            # Timestamp is part of the filename for direct human inspection.
            filename = f"{timestamp:013.6f}.jpg"
            if filename in files:  # defensive guard for repeated decoder PTS
                filename = f"{timestamp:013.6f}-{frame_number:06d}.jpg"
            cv2.imwrite(str(directory / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 88])
            files.append(filename)
            frame_number += 1
            if progress is not None:
                progress.update(1)
            else:
                pending_progress += 1
                if pending_progress >= 16:
                    progress_queue.put((progress_id, pending_progress))
                    pending_progress = 0
    finally:
        cap.release()
        if progress is not None:
            progress.close()
        elif pending_progress:
            progress_queue.put((progress_id, pending_progress))
    data = {"version": 2, "source": str(source.resolve()), "signature": source_signature(source),
            "max_height": max_height, "timestamps": timestamps, "files": files}
    (directory / "index.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return data


def existing_frame_cache(descriptor_dir: Path, source: Path, max_height: int | None = 720) -> tuple[Path, dict[str, Any] | None]:
    directory = cache_path(descriptor_dir, source); index = directory / "index.json"
    if index.exists():
        data = json.loads(index.read_text(encoding="utf-8"))
        if data.get("version") == 2 and data.get("signature") == source_signature(source) and (max_height is None or data.get("max_height") == max_height):
            return directory, data
    return directory, None


def load_frame_cache(descriptor_dir: Path, source: Path, max_height: int | None = 720) -> tuple[Path, dict[str, Any]]:
    directory, data = existing_frame_cache(descriptor_dir, source, max_height)
    return directory, data or build_frame_cache(source, directory, max_height or 720)


def descriptor_caches(d: dict[str, Any], base: Path, max_height: int | None = 720) -> list[tuple[Path, dict[str, Any]]]:
    sources = list(dict.fromkeys((base / tr["source"]).resolve() for tr in d["tracks"]))
    resolved: dict[Path, tuple[Path, dict[str, Any]]] = {}
    missing: list[tuple[Path, Path]] = []
    for source in sources:
        directory, data = existing_frame_cache(base, source, max_height)
        if data is None: missing.append((source, directory))
        else: resolved[source] = (directory, data)
    if missing:
        # OpenCV's decoding, resizing, and JPEG writing happen in native code,
        # so threads can run them concurrently without process startup or IPC
        # overhead. Only this parent thread draws tqdm bars, avoiding garbled
        # cursor sequences from concurrent workers.
        updates: queue.Queue[tuple[int, int]] = queue.Queue()
        bars = [tqdm(total=reported_frame_count(source),
                     desc=f"Cache {source.name}", unit="frame", position=position,
                     leave=True, dynamic_ncols=True)
                for position, (source, _) in enumerate(missing)]
        try:
            with ThreadPoolExecutor(max_workers=len(missing), thread_name_prefix="cache") as pool:
                futures = {pool.submit(build_frame_cache, source, directory, max_height or 720, updates, position): (source, directory, position)
                           for position, (source, directory) in enumerate(missing)}
                while futures:
                    try:
                        while True:
                            position, amount = updates.get_nowait()
                            bars[position].update(amount)
                    except queue.Empty:
                        pass
                    done, _ = wait(futures, timeout=.1, return_when=FIRST_COMPLETED)
                    for future in done:
                        source, directory, position = futures.pop(future)
                        data = future.result()
                        if bars[position].n < len(data["timestamps"]):
                            bars[position].update(len(data["timestamps"]) - bars[position].n)
                        resolved[source] = (directory, data)
                        # Container frame counts are estimates.  Make the
                        # finished line honestly read 100%, even if a decoder
                        # produced one fewer (or more) frame.
                        bars[position].total = bars[position].n
                        bars[position].refresh()
        finally:
            for bar in bars:
                bar.close()
    return [resolved[(base / tr["source"]).resolve()] for tr in d["tracks"]]


def source_frame_index(timestamps: np.ndarray, seconds: float) -> int:
    """Last presented source frame at or before seconds: no interpolation."""
    return max(0, min(len(timestamps) - 1, int(np.searchsorted(timestamps, seconds, side="right") - 1)))


class CachedReader:
    """Fast random-access preview reader backed by predecoded JPEG frames."""
    def __init__(self, directory: Path, cache: dict[str, Any]):
        self.directory, self.timestamps = directory, np.asarray(cache["timestamps"], dtype=np.float64)
        self.source_end = float(self.timestamps[-1])
        self.files: list[str] = cache["files"]
        self.last_index: int | None = None
        self.last_image: np.ndarray | None = None
    def frame(self, sec: float) -> np.ndarray | None:
        index = source_frame_index(self.timestamps, sec)
        if index != self.last_index:
            self.last_image = cv2.imread(str(self.directory / self.files[index]), cv2.IMREAD_COLOR)
            self.last_index = index
        return self.last_image
    def close(self) -> None: pass


class IndexedSourceReader:
    """Full-quality renderer reader that selects frames by cached source PTS."""
    def __init__(self, path: Path, cache: dict[str, Any]):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened(): raise ValueError(f"Cannot open {path}")
        self.timestamps = np.asarray(cache["timestamps"], dtype=np.float64)
        self.source_end = float(self.timestamps[-1])
        self.last_index: int | None = None
        self.last_image: np.ndarray | None = None
    def frame(self, sec: float) -> np.ndarray | None:
        target = source_frame_index(self.timestamps, sec)
        if target == self.last_index:
            return self.last_image
        # During playback the source position normally only moves forward a
        # few frames. Decoding that short run is dramatically faster than a
        # keyframe seek on every preview frame.
        if self.last_index is not None and target > self.last_index and target - self.last_index <= 90:
            img = self.last_image
            for _ in range(target - self.last_index):
                ok, img = self.cap.read()
                if not ok: return None
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, img = self.cap.read()
            if not ok: return None
        self.last_index, self.last_image = target, img
        return img
    def close(self) -> None: self.cap.release()


def transformed_track_frame(d: dict[str, Any], tr: dict[str, Any], reader: Any, t: float, size: tuple[int, int], weight: float) -> np.ndarray | None:
    native_width, native_height = d.get("output", {}).get("size", [1080, 1920])
    width, height = size
    if weight == 0: return None
    img = reader.frame(source_time(d, tr, t))
    if img is None: return None
    if img.shape[:2] == (height, width):
        positioned = img
    else:
        positioned = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    dx, dy = tr.get("translation", [0, 0])
    if dx or dy:
        # Translation is in final-output pixels, so scale it for the smaller
        # preview canvas while keeping renders pixel-accurate.
        matrix = np.float32([[1, 0, dx * width / native_width], [0, 1, dy * height / native_height]])
        positioned = cv2.warpAffine(positioned, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT)
    return positioned.astype(np.float32) * weight


def render_frame(d: dict[str, Any], readers: list[Any], t: float, size: tuple[int, int] | None = None) -> np.ndarray:
    native_width, native_height = d.get("output", {}).get("size", [1080, 1920])
    width, height = size or (native_width, native_height)
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    source_ends = [reader.source_end for reader in readers]
    for tr, reader, weight in zip(d["tracks"], readers, weights(d, t, source_ends)):
        frame = transformed_track_frame(d, tr, reader, t, (width, height), weight)
        if frame is not None: canvas += frame
    return np.uint8(np.clip(canvas, 0, 255))


class TrackRenderWorker:
    """One bounded decode/transform pipeline for a single source track."""
    def __init__(self, d: dict[str, Any], track: dict[str, Any], source: Path, cache: dict[str, Any], size: tuple[int, int], queue_size: int = 1):
        self.d, self.track, self.source, self.cache, self.size = d, track, source, cache, size
        self.tasks: queue.Queue[tuple[int, float, float] | None] = queue.Queue(maxsize=queue_size)
        self.results: queue.Queue[tuple[int, np.ndarray | None, Exception | None]] = queue.Queue(maxsize=queue_size)
        self.ready = threading.Event()
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, name=f"decode-{track['id']}", daemon=True)

    def start(self) -> None:
        self.thread.start(); self.ready.wait()
        if self.error: raise RuntimeError(f"Cannot start track {self.track['id']}") from self.error

    def _run(self) -> None:
        try:
            reader = IndexedSourceReader(self.source, self.cache)
        except Exception as error:
            self.error = error; self.ready.set(); return
        self.ready.set()
        try:
            while (task := self.tasks.get()) is not None:
                index, t, weight = task
                try:
                    self.results.put((index, transformed_track_frame(self.d, self.track, reader, t, self.size, weight), None))
                except Exception as error:
                    self.results.put((index, None, error))
        finally:
            reader.close()

    def submit(self, index: int, t: float, weight: float) -> None:
        self.tasks.put((index, t, weight))

    def result(self, index: int) -> np.ndarray | None:
        result_index, frame, error = self.results.get()
        if error: raise RuntimeError(f"Track {self.track['id']} failed on output frame {index}") from error
        if result_index != index: raise RuntimeError("Track worker returned frames out of order")
        return frame

    def close(self) -> None:
        self.tasks.put(None); self.thread.join()


class VideoEncoder:
    """A bounded writer thread so encoding overlaps the next frame's decoding."""
    def __init__(self, path: Path, fps: float, size: tuple[int, int], queue_size: int = 2):
        self.path, self.fps, self.size = path, fps, size
        self.frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=queue_size)
        self.ready = threading.Event()
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, name="encode", daemon=True)

    def start(self) -> None:
        self.thread.start(); self.ready.wait()
        if self.error: raise RuntimeError("Could not create output video") from self.error

    def _run(self) -> None:
        width, height = self.size
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
        if not writer.isOpened():
            self.error = RuntimeError("VideoWriter failed to open"); self.ready.set(); return
        self.ready.set()
        try:
            while (frame := self.frames.get()) is not None:
                writer.write(frame)
        except Exception as error:
            self.error = error
        finally:
            writer.release()

    def submit(self, frame: np.ndarray) -> None:
        while True:
            if self.error: raise RuntimeError("Encoder failed") from self.error
            try:
                self.frames.put(frame, timeout=.1); return
            except queue.Full: pass

    def close(self) -> None:
        while self.thread.is_alive():
            try:
                self.frames.put(None, timeout=.1)
                break
            except queue.Full:
                if self.error: break
        self.thread.join()
        if self.error: raise RuntimeError("Encoder failed") from self.error


HELP = """Keys: space play/pause | left/right ±1 output frame | up/down ±0.5s | g snap nearest anchor | -/= preview zoom
h add shared hold at playhead | a/d ±1 source frame | A/D ±0.5 source sec | z/x shared hold time ±0.10s
0/1 selected hold opacity set 0/1 | n/p select track | N/P select + solo track | v toggle selected track
i/j/k/l move track 1px | I/J/K/L move 10px
s save | q/esc quit"""


def status_panel(d: dict[str, Any], t: float, selected: int, scale: float, source_ends: list[float] | None = None) -> np.ndarray:
    """Separate, readable status/control panel; never obscures the video."""
    # Six fixed status rows, one row per track, plus the four help rows.
    # Leave a generous bottom margin for font descenders and window scaling.
    panel = np.zeros((max(720, 520 + len(d["tracks"]) * 30), 760, 3), dtype=np.uint8)
    panel[:] = (28, 28, 28)
    hold = min(alignment_holds(d), key=lambda h: abs(h['at'] - t))
    tr = d['tracks'][selected]
    lines = [
        ("CLIMB-CUT — CONTROLS", (80, 220, 120), .78),
        (f"Output: {t:.2f}s / {output_duration(d):.2f}s    Preview zoom: {scale:.0%}", (235, 235, 235), .56),
        (f"Selected: {tr['id']}    {'ON' if tr.get('enabled', True) else 'OFF'}    source: {source_time(d, tr, t):.2f}s", (90, 210, 255), .62),
        (f"Nearest anchor: {hold['id']}  source={anchor_source(tr, hold['id']):.2f}s  opacity={anchor_opacity(tr, hold['id']):.2f}", (90, 210, 255), .50),
        (f"Translation: x={tr.get('translation', [0, 0])[0]:+.0f}px, y={tr.get('translation', [0, 0])[1]:+.0f}px", (90, 210, 255), .56),
        (f"Nearest hold: {hold['id']} at {hold['at']:.2f}s", (235, 235, 235), .56),
        ("Normalised opacity at playhead:", (220, 220, 220), .56),
    ]
    y = 40
    for text, colour, font_scale in lines:
        cv2.putText(panel, text, (22, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, 1, cv2.LINE_AA)
        y += 38
    for index, (track, weight) in enumerate(zip(d['tracks'], weights(d, t, source_ends))):
        colour = (90, 210, 255) if index == selected else (210, 210, 210)
        state = "on" if track.get("enabled", True) else "off"
        cv2.putText(panel, f"{'>' if index == selected else ' '} {track['id']} ({state}): {weight:6.1%}", (30, y), cv2.FONT_HERSHEY_SIMPLEX, .56, colour, 1, cv2.LINE_AA)
        y += 28
    y += 22
    cv2.line(panel, (20, y), (740, y), (100, 100, 100), 1)
    y += 32
    for help_line in HELP.splitlines():
        cv2.putText(panel, help_line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, .48, (235, 235, 235), 1, cv2.LINE_AA)
        y += 34
    return panel


def preview(path: Path, scale: float = .5, cache_height: int = 720) -> None:
    d = load(path); base = path.parent
    caches = descriptor_caches(d, base, cache_height)
    readers = [CachedReader(directory, cache) for directory, cache in caches]
    fps = float(d.get("output", {}).get("fps", 30)); end = output_duration(d)
    width, height = d.get("output", {}).get("size", [1080, 1920])
    t = 0.0; playing = False; selected = 0; last_tick = time.monotonic()
    cv2.namedWindow("climb-cut preview", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("climb-cut controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("climb-cut controls", 760, max(720, 520 + len(d["tracks"]) * 30))
    try:
        while True:
            now = time.monotonic()
            if playing:
                t = min(end, t + now - last_tick)
                if t >= end: playing = False
            last_tick = now
            display_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            # Preview at the display size; render/export still call this at
            # the full output size.
            cv2.imshow("climb-cut preview", render_frame(d, readers, t, display_size))
            source_ends = [reader.source_end for reader in readers]
            cv2.imshow("climb-cut controls", status_panel(d, t, selected, scale, source_ends))
            # waitKeyEx preserves Windows/Qt extended arrow-key codes; waitKey
            # truncates them to zero on some OpenCV builds.
            key = cv2.waitKeyEx(max(1, int(1000/fps) if playing else 30))
            if playing: t = min(end, t + 1/fps)
            if key in (27, ord('q')): break
            if key == ord(' '):
                playing = not playing
                last_tick = time.monotonic()
            if key in (ord('-'), ord('_')): scale = max(.1, round(scale - .1, 2))
            if key in (ord('='), ord('+')): scale = min(1.0, round(scale + .1, 2))
            if key in (81, 2424832, 65361): t = max(0, t - 1 / 30)    # left
            if key in (83, 2555904, 65363): t = min(end, t + 1 / 30)  # right
            if key in (82, 2490368, 65362): t = min(end, t + .5)      # up
            if key in (84, 2621440, 65364): t = max(0, t - .5)        # down
            if key == ord('g'):
                t = min(alignment_holds(d), key=lambda hold: abs(hold['at'] - t))['at']
            if key in (ord('n'), ord('p'), ord('N'), ord('P')):
                selected = (selected + (1 if key in (ord('n'), ord('N')) else -1)) % len(d['tracks'])
                if key in (ord('N'), ord('P')):
                    for index, track in enumerate(d['tracks']):
                        track['enabled'] = index == selected
            if key == ord('v'):
                track = d['tracks'][selected]
                if track.get('enabled', True) and sum(x.get('enabled', True) for x in d['tracks']) == 1:
                    print("At least one track must remain enabled")
                else:
                    track['enabled'] = not track.get('enabled', True)
            hold = min(alignment_holds(d), key=lambda h: abs(h['at']-t))
            if key == ord('h'):
                used = {x['id'] for x in d['holds']}; number = 1
                while f"hold-{number}" in used: number += 1
                mapped_anchors = [(round(source_time(d, tr, t), 3), round(raw_opacity(d, tr, t, reader.source_end), 3)) for tr, reader in zip(d['tracks'], readers)]
                new_hold = {"id": f"hold-{number}", "at": round(t, 3)}
                d['holds'].append(new_hold); d['holds'].sort(key=lambda x: x['at'])
                for tr, (mapped_time, mapped_opacity) in zip(d['tracks'], mapped_anchors):
                    tr['anchors'][new_hold['id']] = {"source": mapped_time, "opacity": mapped_opacity}
                hold = new_hold
            if key in (ord('a'), ord('d')):
                reader = readers[selected]
                track = d['tracks'][selected]
                current = anchor_source(track, hold['id'])
                index = source_frame_index(reader.timestamps, current)
                index = max(0, min(len(reader.timestamps) - 1, index + (-1 if key == ord('a') else 1)))
                set_anchor_source(d, track, hold, round(float(reader.timestamps[index]), 6))
            if key in (ord('A'), ord('D')):
                delta = -.5 if key == ord('A') else .5
                track = d['tracks'][selected]
                set_anchor_source(d, track, hold, round(anchor_source(track, hold['id']) + delta, 3))
            if key in (ord('z'), ord('x')) and hold is not alignment_holds(d)[0]:
                hold_index = d['holds'].index(hold)
                hold['at'] = round(max(d['holds'][hold_index-1]['at']+.01, hold['at'] + (-.1 if key == ord('z') else .1)), 3)
            if key in (ord('0'), ord('1')):
                track = d['tracks'][selected]
                track['anchors'][hold['id']]['opacity'] = int(chr(key))
            if key in (ord('i'), ord('j'), ord('k'), ord('l'), ord('I'), ord('J'), ord('K'), ord('L')):
                amount = 10 if chr(key).isupper() else 1
                dx, dy = d['tracks'][selected].setdefault('translation', [0, 0])
                if key in (ord('j'), ord('J')): dx -= amount
                if key in (ord('l'), ord('L')): dx += amount
                if key in (ord('i'), ord('I')): dy -= amount
                if key in (ord('k'), ord('K')): dy += amount
                d['tracks'][selected]['translation'] = [dx, dy]
            if key == ord('s'):
                try:
                    validate(d)
                    save(path, d)
                    print(f"saved {path}")
                except ValueError as error:
                    print(f"Cannot save: {error}")
    finally:
        for r in readers: r.close()
        cv2.destroyAllWindows()


def render(path: Path, out: Path, single_threaded: bool = False) -> None:
    d = load(path); base = path.parent; fps = 30.0
    width, height = d.get("output", {}).get("size", [1080, 1920])
    out.parent.mkdir(parents=True, exist_ok=True)
    # The PTS index makes frame selection exact even for variable-frame-rate sources.
    caches = descriptor_caches(d, base, None)
    total_frames = math.ceil(output_duration(d) * fps)
    if single_threaded:
        # Reference path for measuring the benefit of the pipelined renderer:
        # all source decoding, transforms, blending, and encoding happen on
        # this thread in output-frame order.
        readers = [IndexedSourceReader((base / track["source"]).resolve(), cache)
                   for track, (_, cache) in zip(d["tracks"], caches)]
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            for reader in readers: reader.close()
            raise RuntimeError("VideoWriter failed to open")
        try:
            with tqdm(total=total_frames, desc="Render output (single thread)", unit="frame", dynamic_ncols=True) as progress:
                for i in range(total_frames):
                    writer.write(render_frame(d, readers, i / fps))
                    progress.update(1)
        finally:
            writer.release()
            for reader in readers: reader.close()
        print(f"\nWrote {out}")
        return

    workers = [TrackRenderWorker(d, track, (base / track["source"]).resolve(), cache, (width, height)) for track, (_, cache) in zip(d["tracks"], caches)]
    encoder = VideoEncoder(out, fps, (width, height))
    try:
        for worker in workers: worker.start()
        encoder.start()
        with tqdm(total=total_frames, desc="Render output", unit="frame", dynamic_ncols=True) as progress:
            for i in range(total_frames):
                t = i / fps
                source_ends = [worker.cache["timestamps"][-1] for worker in workers]
                for worker, weight in zip(workers, weights(d, t, source_ends)):
                    worker.submit(i, t, weight)
                canvas = np.zeros((height, width, 3), dtype=np.float32)
                for worker in workers:
                    frame = worker.result(i)
                    if frame is not None: canvas += frame
                encoder.submit(np.uint8(np.clip(canvas, 0, 255)))
                progress.update(1)
    finally:
        for worker in workers: worker.close()
        encoder.close()
    print(f"\nWrote {out}")


def make_seed(out: Path, videos: list[Path], route: str) -> None:
    info = [metadata(v) for v in videos]
    # Start anchors are placeholders: review them in preview or edit JSON.
    duration = round(max(x[0] for x in info), 3)
    data = {"version": 1, "route": route, "output": {"fps": 30, "size": [1080, 1920], "duration": duration},
      "holds": [{"id":"start", "at":0}], "tracks": []}
    for i, (video, (dur, _, _, _)) in enumerate(zip(videos, info), 1):
        data["tracks"].append({"id":f"attempt-{i}", "source":os.path.relpath(video, out.parent).replace("\\", "/"), "outcome":"unknown", "anchors":{"start":{"source":0, "opacity":1}}, "translation":[0,0], "enabled":True})
    out.parent.mkdir(parents=True, exist_ok=True)
    save(out, data); print(f"Created {out}")


def cache_descriptor(path: Path, height: int) -> None:
    d = load(path)
    descriptor_caches(d, path.parent, height)
    print(f"Cache ready in {path.parent / 'clips'}")


def main() -> None:
    p = argparse.ArgumentParser(prog="climb-cut", description=__doc__)
    sp = p.add_subparsers(dest="command", required=True)
    a = sp.add_parser("init", help="make a route descriptor from attempts"); a.add_argument("descriptor", type=Path); a.add_argument("videos", nargs="+", type=Path); a.add_argument("--route", default="unnamed route")
    command_help = {
        "validate": "check descriptor anchors, translations, and fixed 30 FPS output",
        "preview": "review cached frames and interactively adjust the descriptor",
        "cache": "extract timestamp-named JPEG preview frames into clips/",
    }
    for name in ("validate", "preview", "cache"):
        q = sp.add_parser(name, help=command_help[name], description=command_help[name])
        q.add_argument("descriptor", type=Path)
        if name == "preview":
            q.add_argument("--scale", type=float, default=.5, help="display scale, 0.1 to 1.0 (default: 0.5)")
            q.add_argument("--cache-height", type=int, default=720, help="height of cached preview JPEGs (default: 720)")
        if name == "cache": q.add_argument("--height", type=int, default=720, help="height of JPEG preview frames (default: 720)")
    q = sp.add_parser("render", help="render final fixed-30-FPS MP4 with OpenCV"); q.add_argument("descriptor", type=Path); q.add_argument("--output", type=Path, required=True); q.add_argument("--single-threaded", action="store_true", help="use the sequential reference renderer for benchmarking")
    ns = p.parse_args()
    if ns.command == "init": make_seed(ns.descriptor, ns.videos, ns.route)
    elif ns.command == "validate": load(ns.descriptor); print("Descriptor is valid")
    elif ns.command == "cache": cache_descriptor(ns.descriptor, ns.height)
    elif ns.command == "preview": preview(ns.descriptor, ns.scale, ns.cache_height)
    elif ns.command == "render": render(ns.descriptor, ns.output, ns.single_threaded)

if __name__ == "__main__": main()
