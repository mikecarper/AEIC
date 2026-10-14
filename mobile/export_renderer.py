#!/usr/bin/env python3
"""Export the trained AEIC renderer as small CPU TorchScript/ncnn stages.

No CUDA, entropy decoder, VAE tiling, or model compilation is used. Exporting
does not establish that decoding works on Android: native outputs must still
be compared with reference_rgb.f32.
"""

import argparse
import gc
import importlib.util
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sd-unet-config", type=Path, required=True)
    parser.add_argument("--sd-unet-weights", type=Path, required=True)
    parser.add_argument("--aeic-weights", type=Path, required=True)
    parser.add_argument("--vae-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=tuple(range(128, 577, 64)), default=256)
    parser.add_argument("--adapter", default="auto", help="auto selects student_unet_lora or default")
    parser.add_argument("--lora-alpha", type=float, default=8.0,
                        help="PEFT LoraConfig default used by this repository: 8")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--pnnx", type=Path, help="Optional pnnx executable; otherwise export TorchScript only")
    parser.add_argument("--pnnx-fp16", type=int, choices=(0, 1), default=1,
                        help="ncnn weight-file precision; CPU TorchScript stays FP32")
    parser.add_argument("--pnnx-timeout", type=float, default=300.0)
    parser.add_argument("--l-t", type=Path, help="Optional actual l_T input, raw little-endian float32 NCHW")
    parser.add_argument("--residual", type=Path, help="Optional actual residual input, same format")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--verify-atol", type=float, default=2e-4)
    parser.add_argument("--verify-rtol", type=float, default=2e-4)
    parser.add_argument("--skip-direct-verification", action="store_true")
    parser.add_argument("--resume-conversion", action="store_true",
                        help="Reuse existing ncnn files; only use with unchanged weights and options")
    parser.add_argument("--tiled-attention", action="store_true",
                        help="Use the AEICTiledAttention native layer; enabled automatically at 576")
    parser.add_argument("--reference-only", action="store_true",
                        help="Compute actual-input CPU RGB references without exporting graphs")
    parser.add_argument("--reference-case", nargs=4, action="append", default=[],
                        metavar=("SIZE", "L_T", "RESIDUAL", "OUTPUT_DIR"),
                        help="Additional actual-input reference case; models load only once")
    args = parser.parse_args(argv)
    if bool(args.l_t) != bool(args.residual):
        parser.error("--l-t and --residual must be supplied together")
    if args.threads < 1 or args.pnnx_timeout <= 0:
        parser.error("--threads and --pnnx-timeout must be positive")
    if args.reference_only and not (args.l_t or args.reference_case):
        parser.error("--reference-only requires actual --l-t/--residual or --reference-case inputs")
    if args.reference_case and not args.reference_only:
        parser.error("--reference-case requires --reference-only")
    for size, _, _, _ in args.reference_case:
        if size not in tuple(str(value) for value in range(128, 577, 64)):
            parser.error("reference case size must be a multiple of 64 from 128 to 576")
    return args


def load_checkpoint(torch, path):
    def restricted_lightning_pickle():
        import types
        approved = torch.serialization._weights_only_unpickler._get_allowed_globals()

        class DiscardedCheckpointMetadata:
            def __setstate__(self, state):
                pass

        class RestrictedUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                qualified = module + "." + name
                if qualified == "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint":
                    return DiscardedCheckpointMetadata
                if qualified in approved:
                    return approved[qualified]
                raise pickle.UnpicklingError("Unsupported checkpoint global: " + qualified)

        return types.SimpleNamespace(
            __name__="pickle", Unpickler=RestrictedUnpickler,
            load=lambda stream, **kwargs: RestrictedUnpickler(stream, **kwargs).load(),
        )

    try:
        return torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as error:
        if "mmap" not in str(error):
            raise
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        if "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint" not in str(error):
            raise
        # This published halfDecoder checkpoint has a Lightning callback class
        # as metadata. Discard it with a restricted unpickler instead of
        # importing callback implementations or unrestricted pickle globals.
        return torch.load(str(path), map_location="cpu", weights_only=False,
                          pickle_module=restricted_lightning_pickle(), mmap=True)


def custom_forwards():
    path = Path(__file__).resolve().parents[1] / "src" / "custom_forward.py"
    spec = importlib.util.spec_from_file_location("aeic_mobile_custom_forward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_models(torch, args):
    import types
    from diffusers import UNet2DConditionModel
    try:
        from diffusers.models.autoencoders.vae import Decoder
    except ImportError:
        from diffusers.models.vae import Decoder
    from diffusers.models.attention_processor import AttnProcessor2_0
    from safetensors.torch import load_file

    forwards = custom_forwards()
    config = json.loads(args.sd_unet_config.read_text(encoding="utf-8"))
    # Meta construction avoids allocating random full-model weights beside the
    # supplied checkpoint. assign=True requires PyTorch 2.1 or newer.
    with torch.device("meta"):
        unet = UNet2DConditionModel.from_config(config)
    original = load_file(str(args.sd_unet_weights), device="cpu")
    unet.load_state_dict(original, strict=True, assign=True)
    del original

    for module in unet.modules():
        kind = module.__class__.__name__
        if kind == "ResnetBlock2D":
            if hasattr(module, "time_emb_proj"):
                del module.time_emb_proj
            module.forward = types.MethodType(forwards.MyResnetBlock2D_SD_forward, module)
        elif kind == "BasicTransformerBlock":
            del module.attn2, module.norm2
        elif kind == "Transformer2DModel":
            module.forward = types.MethodType(forwards.MyTransformer2DModel_SD_forward, module)
        elif kind == "CrossAttnDownBlock2D":
            module.forward = types.MethodType(forwards.MyCrossAttnDownBlock2D_SD_forward, module)
        elif kind == "DownBlock2D":
            module.forward = types.MethodType(forwards.MyDownBlock2D_SD_forward, module)
        elif kind == "UNetMidBlock2DCrossAttn":
            module.forward = types.MethodType(forwards.MyUNetMidBlock2DCrossAttn_SD_forward, module)
        elif kind == "CrossAttnUpBlock2D":
            module.forward = types.MethodType(forwards.MyCrossAttnUpBlock2D_SD_forward, module)
        elif kind == "UpBlock2D":
            module.forward = types.MethodType(forwards.MyUpBlock2D_SD_forward, module)
    del unet.time_embedding, unet.time_proj
    unet.forward = types.MethodType(forwards.MyUNet2DConditionModel_SD_forward, unet)
    unet.conv_in = torch.nn.Conv2d(320, 320, 3, padding=1)
    unet.conv_out = torch.nn.Conv2d(320, 256, 3, padding=1)
    unet.float()

    checkpoint = load_checkpoint(torch, args.aeic_weights)
    trained = checkpoint["state_dict_unet"]
    for layer_name in ("conv_in", "conv_out"):
        layer = getattr(unet, layer_name)
        for field in ("weight", "bias"):
            keys = (layer_name + "." + field, "student_" + layer_name + "." + field)
            key = next((item for item in keys if item in trained), None)
            if key is None:
                raise ValueError("AEIC checkpoint lacks " + "/".join(keys))
            with torch.no_grad():
                getattr(layer, field).copy_(trained[key].float())

    adapters = set()
    pairs = []
    for key in trained:
        match = re.match(r"^(.*)\.lora_A\.([^.]+)\.weight$", key)
        if match:
            adapters.add(match[2])
            pairs.append((match[1], match[2], key))
    adapter = args.adapter
    if adapter == "auto":
        adapter = "student_unet_lora" if "student_unet_lora" in adapters else "default"
    if adapter not in adapters:
        raise ValueError("No LoRA adapter %r in checkpoint; found %s" % (adapter, sorted(adapters)))
    merged = 0
    with torch.no_grad():
        for name, candidate, key_a in pairs:
            if candidate != adapter:
                continue
            key_b = key_a.replace(".lora_A.", ".lora_B.")
            if key_b not in trained:
                raise ValueError("Missing paired LoRA tensor " + key_b)
            layer = unet.get_submodule(name)
            a, b = trained[key_a].float(), trained[key_b].float()
            if isinstance(layer, torch.nn.Linear):
                delta = b @ a
            elif isinstance(layer, torch.nn.Conv2d):
                delta = (b.flatten(1) @ a.flatten(1)).view_as(layer.weight)
            else:
                raise TypeError("Unsupported LoRA target " + name)
            layer.weight.add_(delta * (args.lora_alpha / a.shape[0]))
            merged += 1
    del checkpoint, trained, a, b, delta
    gc.collect()
    unet.set_attn_processor(AttnProcessor2_0())

    decoder = Decoder(
        in_channels=4, out_channels=3,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=[64, 128, 256, 256], layers_per_block=2,
        norm_num_groups=32, act_fn="silu", norm_type="group",
        mid_block_add_attention=True,
    )
    vae_checkpoint = load_checkpoint(torch, args.vae_weights)
    source = vae_checkpoint.get("state_dict", vae_checkpoint)
    decoder_weights = {
        key.split("decoder.", 1)[1]: value
        for key, value in source.items() if "decoder." in key
    }
    if not decoder_weights:
        decoder_weights = source
    decoder.load_state_dict(decoder_weights, strict=True)
    expanded = torch.nn.Conv2d(256, 256, 3, padding=1)
    with torch.no_grad():
        expanded.weight.copy_(decoder.conv_in.weight.repeat(1, 64, 1, 1) / 64)
        expanded.bias.copy_(decoder.conv_in.bias)
    decoder.conv_in = expanded
    decoder.forward = types.MethodType(forwards.MyVAEDecoder_SD_forward, decoder)
    for module in decoder.modules():
        if module.__class__.__name__ == "Attention":
            module.set_processor(AttnProcessor2_0())
    del vae_checkpoint, source, decoder_weights
    # Out-of-place activations give the same eval values and make stage tracing
    # safe when a sample tensor is reused by the tracer or residual branches.
    for model in (unet, decoder):
        model.eval().requires_grad_(False)
        for module in model.modules():
            if isinstance(module, (torch.nn.SiLU, torch.nn.Dropout)):
                module.inplace = False
    print("CPU renderer constructed; merged %d %s LoRA targets (alpha=%s)." %
          (merged, adapter, args.lora_alpha), flush=True)
    return unet, decoder, {"adapter": adapter, "lora_alpha": args.lora_alpha,
                           "merged_lora_targets": merged}


def operations(torch):
    nn = torch.nn

    class Unary(nn.Module):
        def __init__(self, operation):
            super().__init__()
            self.operation = operation

        def forward(self, x):
            return self.operation(x)

    class SpatialUnary(nn.Module):
        def __init__(self, operation):
            super().__init__()
            self.operation = operation

        def forward(self, x):
            batch, channels, height, width = x.shape
            # ncnn's three-dimensional Gemm input is Nx1xM, not a
            # batch-by-sequence-by-feature tensor. Remove the known batch-one
            # axis explicitly so conversion emits an ordinary [HW, C] matrix.
            sequence = x.squeeze(0).permute(1, 2, 0).reshape(height * width, channels)
            sequence = self.operation(sequence)
            return sequence.reshape(height, width, sequence.shape[-1]).permute(2, 0, 1).unsqueeze(0).contiguous()

    class Add(nn.Module):
        def __init__(self, factor=1.0):
            super().__init__()
            self.factor = float(factor)

        def forward(self, x, other):
            return (x + other) / self.factor

    class Concat(nn.Module):
        def forward(self, x, other):
            # Native stage inputs are CHW. Make the channel axis explicit
            # instead of retaining the TorchScript batch axis for Concat.
            return torch.cat((x.squeeze(0), other.squeeze(0)), dim=0).unsqueeze(0)

    class Clamp(nn.Module):
        def forward(self, x):
            return x.clamp(-1.0, 1.0)

    class SpatialAttention(nn.Module):
        """Diffusers self-attention expressed with explicit CPU SDPA operations."""

        def __init__(self, attention):
            super().__init__()
            if getattr(attention, "spatial_norm", None) is not None:
                raise ValueError("Spatial conditioning is unsupported in this AEIC exporter")
            self.group_norm = attention.group_norm
            self.to_q = attention.to_q
            self.to_k = attention.to_k
            self.to_v = attention.to_v
            self.out_proj = attention.to_out[0]
            self.dropout = attention.to_out[1]
            self.heads = attention.heads
            self.scale = float(attention.scale)
            self.residual_connection = attention.residual_connection
            self.rescale_output_factor = float(attention.rescale_output_factor)

        def forward(self, x):
            residual = x
            batch, channels, height, width = x.shape
            if self.group_norm is not None:
                x = self.group_norm(x)
            sequence = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
            q = self.to_q(sequence).reshape(batch, height * width, self.heads, -1).transpose(1, 2)
            k = self.to_k(sequence).reshape(batch, height * width, self.heads, -1).transpose(1, 2)
            v = self.to_v(sequence).reshape(batch, height * width, self.heads, -1).transpose(1, 2)
            value = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False, scale=self.scale)
            value = value.transpose(1, 2).reshape(batch, height * width, -1)
            value = self.dropout(self.out_proj(value))
            value = value.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
            if self.residual_connection:
                value = value + residual
            return value / self.rescale_output_factor

    return {name: value for name, value in locals().items()
            if name in ("Unary", "SpatialUnary", "Add", "Concat", "Clamp", "SpatialAttention")}


class Plan:
    def __init__(self, op):
        self.op = op
        self.stages = []
        self.next_slot = 2

    def add(self, name, module, *inputs):
        output = self.next_slot
        self.next_slot += 1
        self.stages.append({"name": name, "module": module.eval(),
                            "input_slots": list(inputs), "output_slot": output})
        return output

    def unary(self, name, module, value, spatial=False):
        wrapper = self.op["SpatialUnary"] if spatial else self.op["Unary"]
        return self.add(name, wrapper(module), value)

    def resnet(self, name, block, value, custom=False):
        residual = value
        value = self.unary(name + ".norm1", block.norm1, value)
        value = self.unary(name + ".act1", block.nonlinearity, value)
        value = self.unary(name + ".conv1", block.conv1, value)
        value = self.unary(name + ".norm2", block.norm2, value)
        value = self.unary(name + ".act2", block.nonlinearity, value)
        value = self.unary(name + ".conv2", block.conv2, value)
        if block.conv_shortcut is not None:
            residual = self.unary(name + ".shortcut", block.conv_shortcut, residual)
        factor = 1.0 if custom else block.output_scale_factor
        return self.add(name + ".add", self.op["Add"](factor), value, residual)

    def transformer(self, name, model, value):
        if model.proj_in.__class__.__name__ != "Linear":
            raise ValueError("Repository custom transformer expects Linear proj_in: " + name)
        residual = value
        value = self.unary(name + ".norm", model.norm, value)
        value = self.unary(name + ".proj_in", model.proj_in, value, spatial=True)
        for index, block in enumerate(model.transformer_blocks):
            stem = name + ".block%d" % index
            normalized = self.unary(stem + ".norm1", block.norm1, value, spatial=True)
            attention = self.add(stem + ".attn", self.op["SpatialAttention"](block.attn1), normalized)
            value = self.add(stem + ".attn_add", self.op["Add"](), value, attention)
            normalized = self.unary(stem + ".norm3", block.norm3, value, spatial=True)
            feedforward = normalized
            for ff_index, layer in enumerate(block.ff.net):
                feedforward = self.unary(stem + ".ff%d" % ff_index, layer, feedforward, spatial=True)
            value = self.add(stem + ".ff_add", self.op["Add"](), value, feedforward)
        value = self.unary(name + ".proj_out", model.proj_out, value, spatial=True)
        return self.add(name + ".add", self.op["Add"](), value, residual)

    def unet(self, model):
        value = self.unary("unet.conv_in", model.conv_in, 0)
        skips = [value]
        for index, block in enumerate(model.down_blocks):
            stem = "unet.down%d" % index
            for res_index, resnet in enumerate(block.resnets):
                value = self.resnet(stem + ".res%d" % res_index, resnet, value, custom=True)
                if hasattr(block, "attentions"):
                    value = self.transformer(stem + ".attn%d" % res_index, block.attentions[res_index], value)
                skips.append(value)
            if block.downsamplers is not None:
                value = self.unary(stem + ".downsample", block.downsamplers[0], value)
                skips.append(value)
        value = self.resnet("unet.mid.res0", model.mid_block.resnets[0], value, custom=True)
        value = self.transformer("unet.mid.attn0", model.mid_block.attentions[0], value)
        value = self.resnet("unet.mid.res1", model.mid_block.resnets[1], value, custom=True)
        for index, block in enumerate(model.up_blocks):
            stem = "unet.up%d" % index
            for res_index, resnet in enumerate(block.resnets):
                value = self.add(stem + ".concat%d" % res_index, self.op["Concat"](), value, skips.pop())
                value = self.resnet(stem + ".res%d" % res_index, resnet, value, custom=True)
                if hasattr(block, "attentions"):
                    value = self.transformer(stem + ".attn%d" % res_index, block.attentions[res_index], value)
            if block.upsamplers is not None:
                value = self.unary(stem + ".upsample", block.upsamplers[0], value)
        if skips:
            raise ValueError("UNet stage schedule left unconsumed skips")
        value = self.unary("unet.output.norm", model.conv_norm_out, value)
        value = self.unary("unet.output.act", model.conv_act, value)
        return self.unary("unet.conv_out", model.conv_out, value)

    def decoder(self, model, value):
        value = self.unary("vae.conv_in", model.conv_in, value)
        value = self.resnet("vae.mid.res0", model.mid_block.resnets[0], value)
        for index, attention in enumerate(model.mid_block.attentions):
            value = self.add("vae.mid.attn%d" % index, self.op["SpatialAttention"](attention), value)
            value = self.resnet("vae.mid.res%d" % (index + 1), model.mid_block.resnets[index + 1], value)
        for index, block in enumerate(model.up_blocks):
            stem = "vae.up%d" % index
            for res_index, resnet in enumerate(block.resnets):
                value = self.resnet(stem + ".res%d" % res_index, resnet, value)
            if block.upsamplers is not None:
                value = self.unary(stem + ".upsample", block.upsamplers[0], value)
        value = self.unary("vae.output.norm", model.conv_norm_out, value)
        value = self.unary("vae.output.act", model.conv_act, value)
        value = self.unary("vae.conv_out", model.conv_out, value)
        return self.add("output.clamp", self.op["Clamp"](), value)

    def releases(self):
        last_use = {}
        for index, stage in enumerate(self.stages):
            for slot in stage["input_slots"]:
                last_use[slot] = index
        for index, stage in enumerate(self.stages):
            stage["release"] = [slot for slot, last in last_use.items() if last == index]


def read_tensor(torch, path, shape):
    import numpy as np
    values = np.fromfile(path, dtype="<f4")
    count = 1
    for dimension in shape:
        count *= dimension
    if values.size != count:
        raise ValueError("%s has %d floats, expected %d" % (path, values.size, count))
    return torch.from_numpy(values.reshape(shape).copy())


def save_tensor(value, path):
    value.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tofile(path)


def reference_only(torch, args, unet, decoder, metadata):
    import numpy as np
    from PIL import Image
    cases = list(args.reference_case)
    if args.l_t:
        cases.insert(0, (args.size, args.l_t, args.residual, args.output))
    with torch.inference_mode():
        for size, latent_path, residual_path, output_path in cases:
            size, output_path = int(size), Path(output_path)
            latent_size = size // 8
            latent = read_tensor(torch, Path(latent_path), [1, 320, latent_size, latent_size])
            residual = read_tensor(torch, Path(residual_path), [1, 256, latent_size, latent_size])
            rgb = decoder(unet(latent) + residual).clamp(-1, 1)
            if not torch.isfinite(rgb).all():
                raise ValueError("Non-finite direct RGB reference for size %d" % size)
            output_path.mkdir(parents=True, exist_ok=True)
            save_tensor(rgb, output_path / "reference_rgb.f32")
            pixels = rgb[0].permute(1, 2, 0).contiguous().numpy()
            pixels = np.clip(np.rint((pixels + 1.0) * 127.5), 0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(output_path / "reference_rgb.png")
            report = {"size": size, "shape": list(rgb.shape), "dtype": "float32",
                      "layout": "NCHW", "rgb_range": [-1, 1], "metadata": metadata,
                      "l_T_input": str(latent_path), "residual_input": str(residual_path),
                      "reference_kind": "CPU renderer on actual supplied codec outputs",
                      "phone_decode_run": False}
            (output_path / "reference.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print("Saved actual-input %d CPU reference: %s" % (size, output_path), flush=True)
            del latent, residual, rgb, pixels


def native_attention_mapping(args, param):
    if not (getattr(args, "tiled_attention", False) or getattr(args, "size", 0) >= 576):
        return
    text = param.read_text(encoding="utf-8")
    mapped = re.sub(r"(?m)^MultiHeadAttention(\s+)", r"AEICTiledAttention\1", text)
    if mapped != text:
        param.write_text(mapped, encoding="utf-8")


def convert(args, filename, shapes):
    if args.pnnx is None:
        return False
    stem = filename.with_suffix("")
    param, binary = Path(str(stem) + ".ncnn.param"), Path(str(stem) + ".ncnn.bin")
    if args.resume_conversion and param.is_file() and binary.is_file():
        native_attention_mapping(args, param)
        return True
    shape_argument = ",".join("[" + ",".join(map(str, shape)) + "]f32" for shape in shapes)
    command = [str(args.pnnx.resolve()), str(filename.resolve()),
               "inputshape=" + shape_argument, "fp16=" + str(args.pnnx_fp16)]
    for key, extension in (("ncnnparam", ".ncnn.param"), ("ncnnbin", ".ncnn.bin"),
                           ("pnnxparam", ".pnnx.param"), ("pnnxbin", ".pnnx.bin"),
                           ("pnnxpy", ".pnnx.py"), ("ncnnpy", ".ncnn.py")):
        command.append(key + "=" + str(Path(str(stem) + extension).resolve()))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=args.pnnx_timeout, check=False)
    Path(str(stem) + ".pnnx.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0 or not param.is_file() or not binary.is_file():
        raise RuntimeError("pnnx failed for %s; inspect %s.pnnx.log" % (filename.name, stem))
    # A converter can return zero while leaving unsupported PNNX operators.
    text = param.read_text(encoding="utf-8")
    unsupported = [line.split()[0] for line in text.splitlines()[2:]
                   if line.strip() and (line.split()[0].startswith(("pnnx.", "aten::", "prim::", "F.")))]
    if unsupported:
        raise RuntimeError("Unsupported converted operators: " + ", ".join(unsupported))
    native_attention_mapping(args, param)
    return True


def shortcut_first_tsv(manifest):
    """Schedule the independent final VAE shortcut early for large images.

    Graph filenames and tensor slots stay identical to the 256 export. Only
    the stage order and last-consumer release rows change.
    """
    from collections import Counter

    stages = manifest["stages"]
    shortcut = next(stage for stage in stages
                    if stage["name"].endswith("_vae_up3_res0_shortcut"))
    norm = next(stage for stage in stages
                if stage["name"].endswith("_vae_up3_res0_norm1"))
    if shortcut["inputs"] != norm["inputs"]:
        raise ValueError("Final VAE shortcut and norm must share their original input")
    stages.remove(shortcut)
    stages.insert(stages.index(norm), shortcut)
    remaining = Counter(item["slot"] for stage in stages for item in stage["inputs"])
    for stage in stages:
        stage["release"] = []
        for item in stage["inputs"]:
            remaining[item["slot"]] -= 1
            if remaining[item["slot"]] == 0:
                stage["release"].append(item["slot"])

    def label(slot):
        if slot == 0:
            return "l_T"
        if slot == 1:
            return "residual"
        return "rgb" if slot == manifest["output_slot"] else "s%05d" % slot

    rows = ["set\toutput_slot\trgb", "set\trgb_range\t-1,1"]
    for item in manifest["inputs"]:
        rows.append("tensor\t%s\t%d\t%d\t%d" % (label(item["slot"]), *item["shape"][1:]))
    for stage in stages:
        for item in stage["outputs"]:
            rows.append("tensor\t%s\t%d\t%d\t%d" % (label(item["slot"]), *item["shape"][1:]))
        rows.append("stage\t%s\t%s\t%s\t%s\t%s" % (
            stage["name"], stage["ncnn_param"], stage["ncnn_bin"],
            ",".join(label(item["slot"]) + "=" + item["name"] for item in stage["inputs"]),
            ",".join(label(item["slot"]) + "=" + item["name"] for item in stage["outputs"])))
        rows.extend("release\t" + label(slot) for slot in stage["release"])
    manifest["shortcut_first_schedule"] = {
        "stage": shortcut["name"], "before": norm["name"],
        "reason": "Release the 128-channel final VAE residual early",
    }
    return rows


def export(torch, args, unet, decoder, metadata):
    op = operations(torch)
    plan = Plan(op)
    prediction = plan.unet(unet)
    latent = plan.add("renderer.residual", op["Add"](), prediction, 1)
    final_slot = plan.decoder(decoder, latent)
    plan.releases()
    latent_size = args.size // 8
    shapes = {0: [1, 320, latent_size, latent_size], 1: [1, 256, latent_size, latent_size]}
    if args.l_t:
        inputs = {0: read_tensor(torch, args.l_t, shapes[0]),
                  1: read_tensor(torch, args.residual, shapes[1])}
        input_kind = "actual codec outputs supplied by caller"
    else:
        generator = torch.Generator().manual_seed(args.seed)
        inputs = {slot: torch.randn(shape, generator=generator) * 0.2
                  for slot, shape in shapes.items()}
        input_kind = "synthetic equivalence test; not an AEIC bitstream decode"
    args.output.mkdir(parents=True, exist_ok=True)
    save_tensor(inputs[0], args.output / "sample_l_T.f32")
    save_tensor(inputs[1], args.output / "sample_residual.f32")
    direct = None
    with torch.inference_mode():
        if not args.skip_direct_verification:
            print("Computing CPU direct reference...", flush=True)
            direct = decoder(unet(inputs[0].clone()) + inputs[1]).clamp(-1, 1)
            save_tensor(direct, args.output / "reference_rgb.f32")
        values = dict(inputs)
        manifest = {"format": "aeic-renderer-stages-v1", "size": args.size,
                    "layout": "NCHW", "dtype": "float32", "input_kind": input_kind,
                    "inputs": [{"slot": slot, "name": "l_T" if slot == 0 else "residual",
                                "shape": shape} for slot, shape in shapes.items()],
                    "output_slot": final_slot, "rgb_range": [-1, 1], "stages": [],
                    "metadata": metadata, "ncnn_weight_file_fp16": bool(args.pnnx_fp16),
                    "native_verification": "not performed by this exporter"}
        if args.tiled_attention or args.size >= 576:
            manifest["native_custom_layer"] = "AEICTiledAttention (all keys, query tiles of 128 rows)"
        tsv = ["set\toutput_slot\trgb", "set\trgb_range\t-1,1"]

        def native_slot(slot):
            if slot == 0:
                return "l_T"
            if slot == 1:
                return "residual"
            return "rgb" if slot == final_slot else "s%05d" % slot

        for slot, shape in shapes.items():
            tsv.append("tensor\t%s\t%d\t%d\t%d" % (native_slot(slot), *shape[1:]))
        for index, stage in enumerate(plan.stages):
            name = "%04d_%s" % (index, stage["name"].replace(".", "_"))
            module = stage["module"]
            stage_inputs = tuple(values[slot] for slot in stage["input_slots"])
            output = module(*stage_inputs).contiguous()
            if output.ndim != 4 or output.shape[0] != 1:
                raise ValueError("Stage boundary must be batch1 NCHW: " + name)
            if not torch.isfinite(output).all():
                raise ValueError("Non-finite CPU stage output: " + name)
            filename = args.output / (name + ".pt")
            traced = torch.jit.trace(module, stage_inputs, check_trace=False, strict=True)
            traced_output = traced(*stage_inputs)
            torch.testing.assert_close(traced_output, output, atol=args.verify_atol, rtol=args.verify_rtol)
            traced.save(str(filename))
            converted = convert(args, filename, [list(tensor.shape) for tensor in stage_inputs])
            record = {"name": name, "torchscript": filename.name,
                      "torchscript_inputs": [value.debugName() for value in list(traced.graph.inputs())[1:]],
                      "torchscript_outputs": [value.debugName() for value in traced.graph.outputs()],
                      "ncnn_param": name + ".ncnn.param", "ncnn_bin": name + ".ncnn.bin",
                      "converted": converted, "release": stage["release"],
                      "parameters": sum(parameter.numel() for parameter in module.parameters()),
                      "inputs": [{"slot": slot, "name": "in%d" % number,
                                  "shape": list(tensor.shape)}
                                 for number, (slot, tensor) in enumerate(zip(stage["input_slots"], stage_inputs))],
                      "outputs": [{"slot": stage["output_slot"], "name": "out0", "shape": list(output.shape)}]}
            manifest["stages"].append(record)
            slot = stage["output_slot"]
            shapes[slot] = list(output.shape)
            values[slot] = traced_output.contiguous()
            tsv.append("tensor\t%s\t%d\t%d\t%d" % (native_slot(slot), *shapes[slot][1:]))
            tsv.append("stage\t%s\t%s\t%s\t%s\t%s" % (
                name, record["ncnn_param"], record["ncnn_bin"],
                ",".join(native_slot(item["slot"]) + "=" + item["name"] for item in record["inputs"]),
                native_slot(slot) + "=out0"))
            for released in stage["release"]:
                values.pop(released, None)
                tsv.append("release\t" + native_slot(released))
            del traced, stage_inputs, output, traced_output
            if index % 20 == 0 or index == len(plan.stages) - 1:
                print("Exported %d/%d: %s" % (index + 1, len(plan.stages), name), flush=True)
            # A partial manifest makes a converter error concrete and resumable.
            (args.output / "renderer_manifest.partial.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staged = values[final_slot]
        save_tensor(staged, args.output / "staged_rgb.f32")
        if direct is not None:
            torch.testing.assert_close(staged, direct, atol=args.verify_atol, rtol=args.verify_rtol)
            error = (staged - direct).abs()
            manifest["cpu_verification"] = {"passed": True, "max_abs_error": error.max().item(),
                                             "mean_abs_error": error.mean().item(),
                                             "atol": args.verify_atol, "rtol": args.verify_rtol}
        else:
            manifest["cpu_verification"] = {"passed": None, "reason": "direct verification explicitly skipped"}
        manifest["output_shape"] = list(staged.shape)
        manifest["maximum_stage_parameters"] = max(stage["parameters"] for stage in manifest["stages"])
        if args.size >= 512:
            tsv = shortcut_first_tsv(manifest)
        (args.output / "renderer_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (args.output / "renderer.tsv").write_text("\n".join(tsv) + "\n", encoding="ascii")
        print(json.dumps({"stages": len(plan.stages), "maximum_stage_parameters": manifest["maximum_stage_parameters"],
                          "cpu_verification": manifest["cpu_verification"],
                          "native_verification": manifest["native_verification"]}), flush=True)
    return manifest


def main(argv=None):
    args = arguments(argv)
    try:
        import torch
    except ImportError as error:
        raise SystemExit("Install CPU torch, diffusers, safetensors and numpy in the host export environment.") from error
    torch.set_num_threads(args.threads)
    unet, decoder, metadata = build_models(torch, args)
    if args.reference_only:
        reference_only(torch, args, unet, decoder, metadata)
    else:
        export(torch, args, unet, decoder, metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
