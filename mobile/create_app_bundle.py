"""Create a verified streaming USTAR FT2 app bundle from eight param packages.

The default writes only a manifest and installation source map. --archive also
writes a streaming archive; shared model weights occur once. No device changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

SIZES = list(range(128, 577, 64))
CHECKPOINT = "519a0f935dc19fa7f8c5d4dd24df32c3d1c017f29834dd6a7bf68587b47a16a1"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    lab = args.lab_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    shared = {}
    blobs = {}
    links = {}
    def small_config(size, directory, name):
        path = directory / name
        if path.is_file():
            return path.read_bytes()
        with tarfile.open(lab / f"packages/codec{size}-params.tar", "r:") as tar:
            return tar.extractfile(f"codec{size}/{name}").read()
    # Read the current promoted graphs, including repaired matrix/concat
    # wrappers. Older transport archives may predate those fixes.
    for size in SIZES:
        codec = (lab / f"models-real-{size}/codec" if size in (256, 512)
                 else lab / f"results/rate-target/codec-{size}")
        renderer = (lab / f"models-{size}/renderer" if size in (256, 512)
                    else lab / f"results/rate-target/renderer-{size}")
        for config, source, count in (("codec.tsv", lab / "models-256/codec", 15),
                                       ("encoder.tsv", lab / "models-real-256/codec", 14),
                                       ("renderer.tsv", renderer, 492)):
            config_dir = renderer if config == "renderer.tsv" else codec
            shape_dir = lab / f"results/codec-shapes{size}"
            if config != "renderer.tsv" and (shape_dir / config).is_file():
                config_dir = shape_dir
            data = small_config(size, config_dir, config)
            blobs[f"codec{size}/{config}"] = data
            stages = [line.split("\t") for line in data.decode().splitlines()
                      if line.startswith("stage\t")]
            if len(stages) != count:
                raise ValueError(f"Incomplete {size} {config}")
            for stage in stages:
                _, name, param, weight, *_ = stage
                if Path(param).name != param or Path(weight).name != weight:
                    raise ValueError("Invalid graph filename")
                blobs[f"codec{size}/{param}"] = (source / param).read_bytes()
                links[f"codec{size}/{weight}"] = f"../shared/{weight}"
                shared.setdefault(weight, source / weight)
        blobs[f"codec{size}/metadata.tsv"] = small_config(size, codec, "metadata.tsv")
        for name in ("z_cdf.bin", "y_cdf.bin", "z_medians.f32"):
            links[f"codec{size}/{name}"] = f"../shared/{name}"
            shared.setdefault(name, lab / "models-256/codec" / name)
    files = []
    sources = {}
    for name, path in sorted(shared.items()):
        target = f"shared/{name}"
        files.append({"path": target, "size": path.stat().st_size, "sha256": digest(path)})
        sources[target] = str(path)
    for target, data in sorted(blobs.items()):
        files.append({"path": target, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        destination = args.output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        sources[target] = str(destination.resolve())
    weights = [f for f in files if f["path"].endswith(".ncnn.bin")]
    weight_text = "".join(f'{f["path"]}\t{f["size"]}\t{f["sha256"]}\n' for f in weights)
    manifest = {"format": "aeic-native-ft2-bundle-v1", "codec_identity": "AEIC_SE_FT2_NCNN",
                "checkpoint_sha256": CHECKPOINT, "resolutions": SIZES,
                "shared_weights_sha256": hashlib.sha256(weight_text.encode("ascii")).hexdigest(),
                "files": files, "links": [{"path": p, "target": t} for p, t in sorted(links.items())]}
    text = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    (args.output / "manifest.json").write_bytes(text)
    (args.output / "installation_sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    if args.archive:
        with tarfile.open(args.archive, "w", format=tarfile.USTAR_FORMAT) as tar:
            from io import BytesIO
            info = tarfile.TarInfo("manifest.json")
            info.size = len(text)
            tar.addfile(info, BytesIO(text))
            for asset in files:
                info = tarfile.TarInfo(asset["path"])
                info.size = asset["size"]
                info.mode = 0o644
                with Path(sources[asset["path"]]).open("rb") as stream:
                    tar.addfile(info, stream)
    print(json.dumps({"manifest": str(args.output / "manifest.json"), "files": len(files),
                      "links": len(links), "bytes": sum(f["size"] for f in files),
                      "shared_weights_sha256": manifest["shared_weights_sha256"]}))


if __name__ == "__main__":
    main()
