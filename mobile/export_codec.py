#!/usr/bin/env python3
"""Export the AEIC entropy decoder and synthesis stages on a CPU host.

The exported phone package has no Python, CUDA, encoder, or CDF-generation
dependency. Raw tensors are little-endian, contiguous, batch-one NCHW. The
native decoder handles rANS, four-part masks, and Gaussian scale indexes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.codec.codec_practical import PixelCodec  # noqa: E402


def array_of(value: Any, dtype: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=np.dtype(dtype))


def write_array(output: Path, name: str, value: Any, dtype: str) -> dict:
    array = array_of(value, dtype)
    array.tofile(output / name)
    return {"file": name, "dtype": np.dtype(dtype).str, "shape": list(array.shape),
            "bytes": int(array.nbytes)}


def write_cdf(output: Path, name: str, cdf: Any, lengths: Any, offsets: Any) -> dict:
    table = array_of(cdf, "<i4")
    sizes = array_of(lengths, "<i4").reshape(-1)
    shifts = array_of(offsets, "<i4").reshape(-1)
    if table.ndim != 2 or len(sizes) != table.shape[0] or len(shifts) != len(sizes):
        raise ValueError(f"Inconsistent CDF dimensions: {name}")
    for row, length in enumerate(sizes):
        if not 2 <= length <= table.shape[1]:
            raise ValueError(f"Invalid CDF length in {name}, row {row}")
        valid = table[row, :length]
        if valid[0] != 0 or valid[-1] != 65536 or not np.all(np.diff(valid) > 0):
            raise ValueError(f"Invalid 16-bit CDF in {name}, row {row}")
    with (output / name).open("wb") as handle:
        handle.write(b"AEICCDF1")
        handle.write(struct.pack("<II", *table.shape))
        handle.write(sizes.tobytes())
        handle.write(shifts.tobytes())
        handle.write(table.tobytes())
    return {"file": name, "magic": "AEICCDF1", "rows": int(table.shape[0]),
            "columns": int(table.shape[1]), "precision": 16}


def load_codec(checkpoint: Path, codec_type: str) -> tuple[PixelCodec, dict]:
    # The public checkpoint is a tensor dictionary; reject arbitrary pickle code.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    trained = saved["state_dict_codec"]
    codec = PixelCodec(codec_type=codec_type).cpu().eval()
    wanted = codec.state_dict()
    missing = sorted(set(wanted) - set(trained))
    if missing:
        raise KeyError(f"Missing codec weights: {missing}")
    # Entropy buffers can be either empty or already materialized in a checkpoint.
    # Resize only registered buffers; neural parameter shape errors remain errors.
    for key, current in wanted.items():
        incoming = trained[key]
        if current.shape != incoming.shape:
            module_path, _, local_name = key.rpartition(".")
            module = codec.get_submodule(module_path)
            if local_name not in module._buffers:
                raise ValueError(f"Parameter shape mismatch: {key}")
            setattr(module, local_name, incoming.detach().cpu().clone())
    codec.load_state_dict({key: trained[key] for key in wanted}, strict=True)
    codec.requires_grad_(False)
    codec.update(force=True)
    return codec, {"loaded_keys": len(wanted),
                   "ignored_checkpoint_keys": sorted(set(trained) - set(wanted))}


def sample_image(image_path: Path | None, size: int) -> tuple[Image.Image, torch.Tensor]:
    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    else:
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        rgb = np.stack((xx / (size - 1), yy / (size - 1),
                        0.5 + 0.25 * np.sin(xx / 11) * np.cos(yy / 17)), axis=-1)
        rgb[size // 4:size // 2, size // 3:2 * size // 3] = (0.95, 0.3, 0.15)
        image = Image.fromarray(np.rint(rgb.clip(0, 1) * 255).astype(np.uint8), "RGB")
    # Match torchvision ToTensor followed by Normalize(.5,.5), including
    # separate float32 rounding at each step, for native encoder comparisons.
    array = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    array = (array - np.float32(0.5)) / np.float32(0.5)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return image, tensor


class ContextStage(torch.nn.Module):
    def __init__(self, codec: PixelCodec, part: int):
        super().__init__()
        self.adapter_in = codec.adapter_in[part]
        self.context = codec.g_c
        self.adapter_out = codec.adapter_out[part]

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        return self.adapter_out(self.context(self.adapter_in(base)))


def validate_and_capture(codec: PixelCodec, image: torch.Tensor, output: Path,
                         include_encoder: bool = False) -> dict:
    """Create a real rANS stream and capture exact original CPU decode tensors."""
    captured_encoding: list[Any] = []
    recorded_decoding: list[dict] = []
    stage_values: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []

    def save_hook(name: str):
        def hook(module, inputs, result):
            stage_values[name] = (inputs[0].detach().clone(), result.detach().clone())
        return hook

    original_encode = codec.entropy_coder.encode_with_z4y_indexes
    original_decode = codec.entropy_coder.decode_stream
    encoder_handles = []
    if include_encoder:
        for number, module in enumerate(codec.g_a.analysis_transform):
            encoder_handles.append(module.register_forward_hook(save_hook(f"g_a{number}")))
        for number, module in enumerate(codec.h_a.reduction):
            encoder_handles.append(module.register_forward_hook(save_hook(f"h_a{number}")))

    def encode_capture(*args):
        captured_encoding[:] = [array_of(arg, "<i2").copy() if i not in (2, 11)
                                else int(arg) for i, arg in enumerate(args)]
        return original_encode(*args)

    def decode_capture(indexes, group):
        result = original_decode(indexes, group)
        recorded_decoding.append({"indexes": array_of(indexes, "<i2").reshape(-1),
                                  "symbols": array_of(result, "<i2").reshape(-1),
                                  "group": int(group)})
        return result

    codec.entropy_coder.encode_with_z4y_indexes = encode_capture
    try:
        with torch.inference_mode():
            codec.entropy_coder.reset()
            codec.compress(image)
            stream = codec.entropy_coder.get_encoded_stream()
    finally:
        codec.entropy_coder.encode_with_z4y_indexes = original_encode
        for handle in encoder_handles:
            handle.remove()
    with (output / "sample.aeic").open("wb") as handle:
        handle.write(struct.pack(">III", image.shape[2], image.shape[3], len(stream)))
        handle.write(stream)

    handles.append(codec.h_s.register_forward_hook(save_hook("h_s")))
    for part in range(4):
        handles.append(codec.adapter_out[part].register_forward_hook(save_hook(f"ctx_out{part}")))
        # ContextStage inputs are the base before adapter_in, not adapter_out inputs.
        handles.append(codec.adapter_in[part].register_forward_pre_hook(
            lambda module, inputs, part=part: stage_values.__setitem__(
                f"ctx_in{part}", (inputs[0].detach().clone(), inputs[0].detach().clone()))))
    for number, module in enumerate(codec.g_s.synthesis_transform):
        handles.append(module.register_forward_hook(save_hook(f"g_s{number}")))
    handles.append(codec.g_s.register_forward_pre_hook(
        lambda module, inputs: stage_values.__setitem__(
            "y_hat", (inputs[0].detach().clone(), inputs[0].detach().clone()))))
    codec.entropy_coder.decode_stream = decode_capture
    y_height, y_width = image.shape[2] // 32, image.shape[3] // 32
    z_size = (1, codec.y_channel // 2, math.ceil(y_height / 4), math.ceil(y_width / 4))
    try:
        with torch.inference_mode():
            codec.entropy_coder.set_stream(stream)
            l_t, residual = codec.decompress(z_size, y_height, y_width)
    finally:
        codec.entropy_coder.decode_stream = original_decode
        for handle in handles:
            handle.remove()
    if len(recorded_decoding) != 5 or len(captured_encoding) != 12:
        raise RuntimeError("Expected one z group and four y groups")

    arrays = {}
    groups = []
    for number, actual in enumerate(recorded_decoding):
        label = "z" if number == 0 else f"y{number - 1}"
        symbol_position = 0 if number == 0 else 3 + 2 * (number - 1)
        index_position = symbol_position + 1
        expected_symbols = captured_encoding[symbol_position].reshape(-1)
        expected_indexes = captured_encoding[index_position].reshape(-1)
        expected_group = captured_encoding[2 if number == 0 else 11]
        # Negative Gaussian indexes transmit no bits and decode to zero.
        expected_symbols = np.where(expected_indexes < 0, 0, expected_symbols).astype("<i2")
        index_mismatch = int(np.count_nonzero(actual["indexes"] != expected_indexes))
        symbol_mismatch = int(np.count_nonzero(actual["symbols"] != expected_symbols))
        groups.append({"name": label, "symbols": len(expected_symbols),
                       "index_mismatches": index_mismatch,
                       "symbol_mismatches": symbol_mismatch,
                       "cdf_group": actual["group"]})
        arrays[f"{label}_indexes"] = write_array(output, f"{label}_indices.i16",
                                                 actual["indexes"], "<i2")
        arrays[f"{label}_symbols"] = write_array(output, f"{label}_symbols.i16",
                                                 actual["symbols"], "<i2")
        arrays[f"encode_{label}_indexes"] = write_array(
            output, f"encode_{label}_indices.i16", expected_indexes, "<i2")
        arrays[f"encode_{label}_symbols"] = write_array(
            output, f"encode_{label}_symbols.i16",
            captured_encoding[symbol_position].reshape(-1), "<i2")
        if index_mismatch or symbol_mismatch or actual["group"] != expected_group:
            raise RuntimeError(f"Original CPU rANS roundtrip failed for {label}: {groups[-1]}")

    stages = {"h_s": stage_values["h_s"]}
    for part in range(4):
        stages[f"ctx{part}"] = (stage_values[f"ctx_in{part}"][0],
                                stage_values[f"ctx_out{part}"][1])
    for number in range(len(codec.g_s.synthesis_transform)):
        stages[f"g_s{number}"] = stage_values[f"g_s{number}"]
    arrays["z_hat"] = write_array(output, "z_hat.f32", stages["h_s"][0], "<f4")
    arrays["y_hat"] = write_array(output, "y_hat.f32", stage_values["y_hat"][0], "<f4")
    arrays["l_T"] = write_array(output, "l_T.f32", l_t, "<f4")
    arrays["residual"] = write_array(output, "residual.f32", residual, "<f4")
    encoder_stages = {name: pair for name, pair in stage_values.items()
                      if name.startswith("g_a") or name.startswith("h_a")}
    return {"stages": stages, "arrays": arrays, "groups": groups,
            "encoder_stages": encoder_stages,
            "stream_bytes": len(stream), "sample_bytes": len(stream) + 12}


def export_stage(name: str, module: torch.nn.Module, reference: tuple,
                 input_slot: str, output_slot: str, output: Path,
                 pnnx: str | None) -> dict:
    example, expected = reference
    path = output / f"{name}.pt"
    with torch.inference_mode():
        traced = torch.jit.trace(module.eval(), example, check_trace=True)
        traced = torch.jit.freeze(traced)
        actual = traced(example)
        error = float((actual - expected).abs().max())
        if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-5):
            raise RuntimeError(f"TorchScript stage {name} differs from original CPU output: {error}")
        torch.jit.save(traced, path)
    write_array(output, f"{name}.{output_slot}.f32", expected, "<f4")
    write_array(output, f"{name}.input.f32", example, "<f4")
    entry = {"name": name, "torchscript": path.name,
             "ncnn_param": f"{name}.ncnn.param", "ncnn_bin": f"{name}.ncnn.bin",
             "inputs": [{"slot": input_slot, "name": "in0", "shape": list(example.shape)}],
             "outputs": [{"slot": output_slot, "name": "out0", "shape": list(expected.shape)}],
             "torchscript_max_abs_error": error,
             "parameter_count": sum(p.numel() for p in module.parameters())}
    if pnnx:
        shape = "[" + ",".join(str(v) for v in example.shape) + "]f32"
        command = [pnnx, str(path), f"inputshape={shape}", "fp16=0",
                   f"ncnnparam={output / entry['ncnn_param']}",
                   f"ncnnbin={output / entry['ncnn_bin']}",
                   f"pnnxparam={output / (name + '.pnnx.param')}",
                   f"pnnxbin={output / (name + '.pnnx.bin')}",
                   f"pnnxpy={output / (name + '.pnnx.py')}" ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (output / f"{name}.pnnx.log").write_text(result.stdout, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"pnnx conversion failed for {name}; see {name}.pnnx.log")
        if not (output / entry["ncnn_param"]).is_file() or not (output / entry["ncnn_bin"]).is_file():
            raise RuntimeError(f"pnnx did not produce ncnn files for {name}")
    print(f"Exported {name}: {list(example.shape)} -> {list(expected.shape)}; error={error:.3g}", flush=True)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--codec-type", choices=("AEIC-SE", "AEIC-ME"), default="AEIC-SE")
    parser.add_argument("--size", type=int, choices=tuple(range(128, 577, 64)), default=256)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--pnnx", help="pnnx executable; omit to export TorchScript only")
    parser.add_argument("--include-encoder", action="store_true",
                        help="Also export g_a/h_a and exact encoder-stage references")
    parser.add_argument("--encoder-only", action="store_true",
                        help="Export only g_a/h_a graphs while refreshing the sample references")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    codec, load_info = load_codec(args.checkpoint, args.codec_type)
    image, normalized = sample_image(args.image, args.size)
    image.save(output / "sample.png")
    image.save(output / "input.ppm")
    write_array(output, "sample.input.f32", normalized, "<f4")
    cdfs = {
        "z": write_cdf(output, "z_cdf.bin", codec.z_quantized_cdf,
                       codec.z_cdf_length, codec.z_offset),
        "y": write_cdf(output, "y_cdf.bin", codec.y_quantized_cdf,
                       codec.y_cdf_length, codec.y_offset),
    }
    medians = write_array(output, "z_medians.f32",
                          codec.entropy_bottleneck.my_medians.reshape(-1), "<f4")
    capture = validate_and_capture(codec, normalized, output,
                                   args.include_encoder or args.encoder_only)
    stages = []
    if args.encoder_only:
        previous = output / "codec.json"
        if previous.is_file():
            stages = json.loads(previous.read_text(encoding="utf-8"))["stages"]
    else:
        stages.append(export_stage("h_s", codec.h_s, capture["stages"]["h_s"],
                                   "z_hat", "base", output, args.pnnx))
        for part in range(4):
            name = f"ctx{part}"
            stages.append(export_stage(name, ContextStage(codec, part), capture["stages"][name],
                                       "base", "means_scales", output, args.pnnx))
        for number, module in enumerate(codec.g_s.synthesis_transform):
            name = f"g_s{number}"
            first = number == 0
            last = number == len(codec.g_s.synthesis_transform) - 1
            stages.append(export_stage(name, module, capture["stages"][name],
                                       "y_hat" if first else "synth",
                                       "synthesis" if last else "synth", output, args.pnnx))
    encoder_stages = []
    if args.include_encoder or args.encoder_only:
        for prefix, sequence, first_slot, temporary_slot, final_slot in (
            ("g_a", codec.g_a.analysis_transform, "x", "analysis", "y"),
            ("h_a", codec.h_a.reduction, "y", "hyper", "z"),
        ):
            for number, module in enumerate(sequence):
                name = f"{prefix}{number}"
                encoder_stages.append(export_stage(
                    name, module, capture["encoder_stages"][name],
                    first_slot if number == 0 else temporary_slot,
                    final_slot if number == len(sequence) - 1 else temporary_slot,
                    output, args.pnnx))
    manifest = {
        "format": "aeic-codec-stages-v1", "codec_type": args.codec_type,
        "input_height": args.size, "input_width": args.size, "y_channels": codec.y_channel,
        "raw_tensor_layout": "little-endian float32 NCHW, batch=1",
        "sample_bitstream": "sample.aeic", "sample_source": "sample.png",
        "bitstream_header": "three big-endian uint32: height,width,payload_length",
        "stream_parts": 2, "cdfs": cdfs, "z_medians": medians,
        "scale_index": {"minimum": 0.11, "maximum": 256.0, "levels": 64,
                        "zero_symbol_below": 0.08, "rounding": "truncate toward zero"},
        "partition_order": [[0, 1, 2, 3], [3, 2, 1, 0], [2, 3, 0, 1], [1, 0, 3, 2]],
        "stages": stages, "references": capture["arrays"],
        "encoder_stages": encoder_stages,
        "validation": {"original_cpu_rans": capture["groups"],
                       "stream_bytes": capture["stream_bytes"],
                       "sample_file_bytes": capture["sample_bytes"], **load_info},
    }
    (output / "codec.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = ["\t".join(("stage", stage["name"], stage["ncnn_param"], stage["ncnn_bin"],
                         ",".join(f"{item['slot']}={item['name']}" for item in stage["inputs"]),
                         ",".join(f"{item['slot']}={item['name']}" for item in stage["outputs"])))
             for stage in stages]
    lines += ["\t".join(("stage_shape", stage["name"], item["slot"],
                           *(str(value) for value in item["shape"][1:])))
              for stage in stages for item in stage["outputs"]]
    (output / "codec.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if encoder_stages:
        encoder_lines = ["\t".join(("stage", stage["name"], stage["ncnn_param"], stage["ncnn_bin"],
                                     ",".join(f"{item['slot']}={item['name']}" for item in stage["inputs"]),
                                     ",".join(f"{item['slot']}={item['name']}" for item in stage["outputs"])))
                         for stage in encoder_stages]
        encoder_lines += ["\t".join(("stage_shape", stage["name"], item["slot"],
                                    *(str(value) for value in item["shape"][1:])))
                           for stage in encoder_stages for item in stage["outputs"]]
        (output / "encoder.tsv").write_text("\n".join(encoder_lines) + "\n", encoding="utf-8")
    metadata = f"y_channels\t{codec.y_channel}\ninput_height\t{args.size}\ninput_width\t{args.size}\n"
    (output / "metadata.tsv").write_text(metadata, encoding="utf-8")
    print(json.dumps(manifest["validation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
