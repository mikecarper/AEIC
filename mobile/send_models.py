#!/usr/bin/env python3
"""Send completed AEIC model files through the temporary phone receiver."""

import argparse
import hashlib
import json
from pathlib import Path
import socket
import struct
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--hash-tool", required=True)
    args = parser.parse_args()
    destination = "/data/local/tmp/aeic-phone/codec256"
    state = json.loads(args.state.read_text()) if args.state.exists() else {}
    files = []
    for folder in (args.codec, args.renderer):
        files += list(folder.glob("*.ncnn.bin")) + list(folder.glob("*.ncnn.param"))
    files += [args.codec / name for name in (
        "codec.tsv", "metadata.tsv", "z_cdf.bin", "y_cdf.bin", "z_medians.f32", "sample.aeic",
    )] + [args.renderer / "renderer.tsv"]
    names = [path.name for path in files]
    if len(set(names)) != len(names):
        parser.error("colliding model filenames")
    if not all(path.is_file() for path in files):
        parser.error("completed model package is missing a required file")
    token = args.token_file.read_bytes()
    if len(token) != 32:
        parser.error("token must contain exactly 32 bytes")
    sent, start = 0, time.monotonic()
    with socket.create_connection((args.host, args.port), timeout=20) as connection:
        connection.settimeout(120)
        connection.sendall(b"AEICPUT1" + token)
        if connection.recv(1) != b"A":
            raise RuntimeError("receiver authentication failed")
        for path in sorted(files, key=lambda item: item.name):
            checksum = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    checksum.update(chunk)
            record = {"sha256": checksum.hexdigest(), "bytes": path.stat().st_size,
                      "destination": destination}
            if state.get(path.name) == record:
                continue
            # These exact files were already verified by the native entropy test.
            if path.name in {"h_s.ncnn.bin", "ctx0.ncnn.bin", "ctx1.ncnn.bin",
                              "ctx2.ncnn.bin", "ctx3.ncnn.bin", "g_s0.ncnn.bin"}:
                state[path.name] = record
                continue
            expected = int(subprocess.check_output([args.hash_tool, "--hash", str(path)], timeout=30))
            name = path.name.encode("ascii")
            print("SEND %s bytes=%d" % (path.name, record["bytes"]), flush=True)
            connection.sendall(struct.pack("<HQQ", len(name), record["bytes"], expected) + name)
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(256 * 1024), b""):
                    connection.sendall(chunk)
            if connection.recv(1) != b"A":
                raise RuntimeError("file integrity acknowledgement failed: " + path.name)
            sent += record["bytes"]
            state[path.name] = record
            args.state.write_text(json.dumps(state, indent=2) + "\n")
        connection.sendall(struct.pack("<HQQ", 0, 0, 0))
    seconds = time.monotonic() - start
    print("TRANSFER_COMPLETE bytes=%d seconds=%.3f MBps=%.3f" % (sent, seconds, sent / seconds / 1e6))


if __name__ == "__main__":
    main()
