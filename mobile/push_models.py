#!/usr/bin/env python3
"""Resume model transfers to a verified phone without sending training files."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--expected-phone", required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--renderer-cutoff", type=int)
    parser.add_argument("--skip-name", action="append", default=[])
    args = parser.parse_args()
    target = PurePosixPath(args.destination)
    if ".." in target.parts or not str(target).startswith("/data/local/tmp/aeic-phone/"):
        parser.error("destination must be an isolated directory below /data/local/tmp/aeic-phone/")
    if not re.fullmatch(r"/[A-Za-z0-9_/-]+", str(target)):
        parser.error("invalid destination path")
    command = [args.adb, "-s", args.serial]
    identity = subprocess.check_output(command + ["shell", "getprop", "ro.serialno"], timeout=20)
    if identity.decode().strip() != args.expected_phone:
        parser.error("connected phone identity does not match --expected-phone")
    subprocess.run(command + ["shell", "mkdir", "-p", str(target)], check=True, timeout=20)
    state = json.loads(args.state.read_text()) if args.state.exists() else {}
    files = list(args.codec.glob("*.ncnn.param")) + list(args.codec.glob("*.ncnn.bin"))
    files += [args.codec / name for name in (
        "codec.tsv", "metadata.tsv", "z_cdf.bin", "y_cdf.bin", "z_medians.f32", "sample.aeic",
    )]
    for pattern in ("*.ncnn.param", "*.ncnn.bin"):
        for path in args.renderer.glob(pattern):
            match = re.match(r"^(\d+)_", path.name)
            if args.renderer_cutoff is None or (match and int(match[1]) < args.renderer_cutoff):
                files.append(path)
    if args.renderer_cutoff is None:
        files.append(args.renderer / "renderer.tsv")
    names = set()
    sent = 0
    for path in sorted(files, key=lambda item: item.name):
        if path.name in names:
            parser.error("colliding model filenames: " + path.name)
        names.add(path.name)
        if path.name in args.skip_name:
            continue
        if not path.is_file():
            parser.error("required completed model file missing: " + str(path))
        checksum = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                checksum.update(chunk)
        digest = checksum.hexdigest()
        record = {"sha256": digest, "bytes": path.stat().st_size, "destination": str(target)}
        if state.get(path.name) == record:
            continue
        print("PUSH %s bytes=%d" % (path.name, record["bytes"]), flush=True)
        subprocess.run(command + ["push", str(path), str(target / path.name)], check=True, timeout=1800)
        state[path.name] = record
        args.state.write_text(json.dumps(state, indent=2) + "\n")
        sent += record["bytes"]
    print("TRANSFER_COMPLETE sent_bytes=%d tracked_files=%d" % (sent, len(state)), flush=True)


if __name__ == "__main__":
    main()
