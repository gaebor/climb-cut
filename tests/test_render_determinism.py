"""Regression test: threaded and sequential render paths compose identically."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "tests" / "artifacts"
sys.path.insert(0, str(ROOT))
from climb_cut import render  # noqa: E402


SIZE = (64, 48)  # width, height; deliberately tiny to keep this test fast.
FPS = 30.0
FRAME_COUNT = 18


def write_source(path: Path, base: tuple[int, int, int], phase: int) -> None:
    """Make a deterministic source clip with motion and per-frame detail."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    if not writer.isOpened():
        raise RuntimeError("OpenCV VideoWriter failed to open test source")
    width, height = SIZE
    try:
        for i in range(FRAME_COUNT):
            image = np.full((height, width, 3), base, dtype=np.uint8)
            x = (i * 5 + phase) % (width - 12)
            cv2.rectangle(image, (x, 8), (x + 11, 29), (20 + i * 7, 220 - i * 5, 60 + phase), -1)
            cv2.putText(image, str(i), (2, 44), cv2.FONT_HERSHEY_SIMPLEX, .32, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(image)
    finally:
        writer.release()


def decoded_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return frames
            frames.append(frame)
    finally:
        cap.release()


class RenderDeterminismTest(unittest.TestCase):
    def test_threaded_and_sequential_frames_match(self) -> None:
        # Keep fixtures and results around for visual inspection after tests.
        # They are generated files and ignored by Git.
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        write_source(ARTIFACTS / "first.mp4", (25, 70, 190), 0)
        write_source(ARTIFACTS / "second.mp4", (180, 40, 35), 9)
        descriptor = {
            "version": 1,
            "route": "determinism-test",
            "output": {"fps": 30, "size": list(SIZE), "duration": 0.5},
            "holds": [{"id": "start", "at": 0}, {"id": "blend", "at": 0.25}],
            "tracks": [
                {"id": "first", "source": "first.mp4", "outcome": "unknown", "enabled": True,
                 "translation": [3, -2], "anchors": {
                     "start": {"source": 0, "opacity": 1},
                     "blend": {"source": 0.25, "opacity": 0.2},
                 }},
                {"id": "second", "source": "second.mp4", "outcome": "unknown", "enabled": True,
                 "translation": [-2, 1], "anchors": {
                     "start": {"source": 0.05, "opacity": 0.3},
                     "blend": {"source": 0.3, "opacity": 1},
                 }},
            ],
        }
        route = ARTIFACTS / "route.json"
        route.write_text(json.dumps(descriptor), encoding="utf-8")
        sequential, threaded = ARTIFACTS / "sequential.mp4", ARTIFACTS / "threaded.mp4"

        render(route, sequential, single_threaded=True)
        render(route, threaded)

        expected, actual = decoded_frames(sequential), decoded_frames(threaded)
        self.assertEqual(len(expected), 15)
        self.assertEqual(len(actual), len(expected))
        for index, (left, right) in enumerate(zip(expected, actual)):
            np.testing.assert_array_equal(left, right, err_msg=f"output frame {index} differs")


if __name__ == "__main__":
    unittest.main()
