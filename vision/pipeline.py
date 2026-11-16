import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

import cv2
import numpy as np

STAGES = ("load", "grayscale", "threshold", "write")


def threshold(gray: np.ndarray, cutoff: int = 160) -> np.ndarray:
    """Set pixels of intensity at least cutoff to 255 and all others to zero."""
    mask = np.zeros_like(gray)
    mask[gray >= cutoff] = 255
    return mask


def write_results(mask: np.ndarray, directory: Path) -> None:
    if not cv2.imwrite(str(directory / "mask.png"), mask):
        raise OSError(f"Cannot write mask: {directory / 'mask.png'}")

    height, width = mask.shape
    stats = {
        "width": width,
        "height": height,
        "selected_pixels": int(np.count_nonzero(mask)),
        "mask_pixels_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
    }
    (directory / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


def run(image: Path, directory: Path) -> dict[str, float]:
    start = perf_counter_ns()
    color = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if color is None:
        raise OSError(f"Cannot open image: {image}")
    loaded = perf_counter_ns()
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    converted = perf_counter_ns()
    mask = threshold(gray)
    thresholded = perf_counter_ns()
    write_results(mask, directory)
    finished = perf_counter_ns()

    boundaries = (start, loaded, converted, thresholded, finished)
    timings = {
        stage: (boundaries[index + 1] - boundaries[index]) / 1e6
        for index, stage in enumerate(STAGES)
    }
    timings["total"] = (finished - start) / 1e6
    return timings
