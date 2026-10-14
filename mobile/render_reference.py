#!/usr/bin/env python3
"""Render a batch-one CHW float32 decoder output as a PNG."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, choices=(256, 512), required=True)
    args = parser.parse_args()
    expected_bytes = 3 * args.size * args.size * 4
    if args.input.stat().st_size != expected_bytes:
        parser.error("input size does not match three float32 image channels")
    image = np.fromfile(args.input, dtype="<f4").reshape(3, args.size, args.size)
    if not np.isfinite(image).all():
        parser.error("input contains nonfinite values")
    image = np.clip(image * 0.5 + 0.5, 0, 1)
    pixels = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(args.output)
    print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
