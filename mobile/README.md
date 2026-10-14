# AEIC on Android: native CLI prototype

This prototype runs AEIC-SE encoding and decoding with native ncnn CPU inference
on ARMv7 NEON, Android API 21 or later. The tested phone is an LG L51AL running
Android 5.1.1 / API 22 with 1 GB RAM. It is a command line experiment accessed
through ADB.

The supported square input sizes are **128, 192, 256, 320, 384, 448, 512, and
576 pixels**. Export a renderer parameter set for the chosen size and use its
matching metadata. The 64- and 1024-pixel experiments are research artifacts
outside this production allowlist.

The decoder has roughly 914 million parameters. Exported inference weights take
about **2 GB of phone storage**. The runtime loads, executes, and releases one
graph at a time, keeping the complete model out of RAM. Entropy and encoder
weights remain FP32; renderer files use FP16 storage with FP32 runtime arithmetic.

## Verified results

Measured on 2026-10-13 using the real Pixnio photograph below. The earlier
256/512 encoder runs used two threads; the other sizes used four threads.

| Input | Bitstream, including header | Bits/pixel | Encoding proof | Phone encode time | Phone encode peak RSS |
|---|---:|---:|---|---:|---:|
| 128 × 128 | 137 bytes | 0.06689 | Phone and original CPU codec | 2.172 s | 45.9 MiB |
| 192 × 192 | 270 bytes | 0.05859 | Phone and original CPU codec | 2.934 s | 46.0 MiB |
| 256 × 256 | 407 bytes | 0.04968 | Phone and original CPU codec | 6.213 s | 48.9 MiB |
| 320 × 320 | 602 bytes | 0.04703 | Phone and original CPU codec | 3.889 s | 47.3 MiB |
| 384 × 384 | 804 bytes | 0.04362 | Phone and original CPU codec | 5.075 s | 49.4 MiB |
| 448 × 448 | 1,122 bytes | 0.04472 | Phone and original CPU codec | 6.630 s | 64.0 MiB |
| 512 × 512 | 1,363 bytes | 0.04160 | Phone and original CPU codec | 13.632 s | 148.0 MiB |
| 576 × 576 | 1,783 bytes | 0.04299 | Phone and original CPU codec | 10.348 s | 100.0 MiB |

All eight phone-generated files are byte-identical to the original CPU codec's files.
The phone decoded its own entropy streams with zero index or symbol mismatches
and valid terminal rANS states. Complete normal RGB reconstruction has passed
on the phone at these sizes:

| Input | Full decode time | Peak RSS |
|---|---:|---:|
| 128 × 128 | 114.477 s | 236.5 MiB |
| 256 × 256 | 367.101 s | 242.7 MiB |
| 512 × 512 | 1,664.579 s | 550.5 MiB |

The 128-pixel run produced a 45,610-byte PNG; the 256-pixel run produced a
171,241-byte PNG. The 256-pixel PNG differs from the full-precision CPU reference
by 0.579 RGB8 levels on average (48.64 dB PSNR). This is a lossy reconstruction;
its PSNR against the scaled original is 16.12 dB. At 512 pixels, mean error
against the CPU reference is 0.505 RGB8 levels (50.618 dB PSNR). Full 576-pixel
phone reconstruction is pending.

All eight CPU files pass exact entropy round-trip checks. The intermediate
sizes and 576 pixels still need full phone RGB validation. File size depends
on image content: accepting a resolution in this range does not guarantee a
2,000-byte or 2,048-byte file budget. The measured 576-pixel file fits both
limits for this photograph.

The [CC0 example and benchmark artifacts](examples/pixnio/README.md) provide
reviewable inputs, streams, and reconstructions. Their tiny streams require
the matching approximately 2 GB trained model to decode; these file sizes are
measurements of this photograph.

Optional `--best-effort` recovery isolates model stages and labels substituted
results `DEGRADED`. A damaged 128-pixel test with reference and model failures
completed in 137.142 seconds with six recovery events and a 44,449-byte PNG.
Fallback can produce a usable image but cannot restore lost details.

## Host export

Run from the repository root on a Linux host with Python 3.10, CMake, a C++
compiler, Git, and enough RAM/disk space for model conversion. Tested versions:

```sh
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==1.26.4 diffusers==0.25.1 peft==0.13.2 \
  transformers==4.46.3 huggingface_hub==0.25.0 safetensors==0.8.0 \
  scipy==1.15.3 Pillow==12.3.0 pnnx==20260526
cmake -S src/cpp -B build/entropy-host \
  -DPYTHON_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_CXX_FLAGS=-Wno-error=maybe-uninitialized
cmake --build build/entropy-host -j2
```

The original CMake build copies `MLCodec_CXX` and `MLCodec_rans` Python extensions
into `src/codec`. Obtain these pretrained files separately:

- `AEIC_SE_ft2.pkl` from the [authors' checkpoint folder](https://drive.google.com/drive/folders/1vioCW4EIxQiuLkWHKj7xbMi7WAVcWqJI?usp=drive_link).
- `unet/config.json` and `unet/diffusion_pytorch_model.fp16.safetensors` from [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo/tree/main/unet).
- `halfDecoder.ckpt` from [AdcSR](https://huggingface.co/Guaishou74851/AdcSR/blob/main/weight/pretrained/halfDecoder.ckpt).

The commands below place the two checkpoints in `weights/`, and save the UNet
files there as `unet_config.json` and `unet_fp16.safetensors`.

No training dataset, text encoder, tokenizer, scheduler, or full SD-Turbo VAE is
needed by these exporters. The tensor checkpoints are loaded with restricted
deserialization; the published VAE's unused Lightning callback metadata is discarded.

Prepare a square RGB image before exporting. The tested source is Pixnio's
[river/forest photograph](https://pixnio.com/nature-landscapes/river/landscape-water-nature-tree-river-forest-mountain),
listed as [CC0](https://creativecommons.org/publicdomain/zero/1.0/). Its
[1100 × 733 JPEG](https://pixnio.com/free-images/2017/09/22/2017-09-22-08-14-44-1100x733.jpg)
has SHA-256 `206d4ef2885d0fc4a23aa066141c5a31aa415381446fa008a82c515f7577f275`.
Save this source JPEG as `inputs/river.jpg` before running the preparation code.
The benchmark resized with LANCZOS, then cropped the center square. For
128/256/512/576 pixels, the intermediate sizes were 192 × 128, 384 × 256,
768 × 512, and 864 × 576; the corresponding left crop offsets were 32, 64,
128, and 144 pixels.

```python
from pathlib import Path
from PIL import Image, ImageOps

image = ImageOps.exif_transpose(Image.open("inputs/river.jpg")).convert("RGB")
for size in (128, 256, 512, 576):
    scale = size / min(image.size)
    dimensions = tuple(max(size, round(value * scale)) for value in image.size)
    scaled = image.resize(dimensions, Image.Resampling.LANCZOS)
    square = ImageOps.fit(scaled, (size, size), method=Image.Resampling.LANCZOS,
                         centering=(0.5, 0.5))
    square.save(Path("inputs") / f"river-{size}.png")
```

Export one resolution; adjust `--size` and paths for the other:

```sh
python mobile/export_codec.py --checkpoint weights/AEIC_SE_ft2.pkl \
  --output-dir build/models-256/codec --image inputs/river-256.png \
  --size 256 --threads 2 --include-encoder --pnnx "$(command -v pnnx)"
python mobile/export_renderer.py \
  --sd-unet-config weights/unet_config.json \
  --sd-unet-weights weights/unet_fp16.safetensors \
  --aeic-weights weights/AEIC_SE_ft2.pkl --vae-weights weights/halfDecoder.ckpt \
  --output build/models-256/renderer --size 256 --threads 2 \
  --l-t build/models-256/codec/l_T.f32 \
  --residual build/models-256/codec/residual.f32 --pnnx "$(command -v pnnx)"
```

The codec exporter produces a valid sample stream, `input.ppm`, CDF tables,
encoder/decoder TSV manifests, and exact CPU reference tensors. It verifies the
original CPU entropy round-trip and TorchScript stage outputs. Native ncnn
numerical agreement requires the separate phone checks below.

## Native build and deployment

The tested ncnn revision is
`b0f97f43f266617ff5e5096d440dbe3f9a4c35b8`; Android NDK version was
`29.0.14206865`. Set `ANDROID_NDK` to its installation directory.

```sh
git clone https://github.com/Tencent/ncnn.git tools/ncnn
git -C tools/ncnn checkout b0f97f43f266617ff5e5096d440dbe3f9a4c35b8
cmake -S tools/ncnn -B build/ncnn-android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=armeabi-v7a -DANDROID_PLATFORM=android-21 \
  -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/ncnn-install" \
  -DNCNN_VULKAN=OFF -DNCNN_SHARED_LIB=OFF -DNCNN_RUNTIME_CPU=ON \
  -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF
cmake --build build/ncnn-android -j2
cmake --install build/ncnn-android
cmake -S mobile/native -B build/phone \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=armeabi-v7a -DANDROID_PLATFORM=android-21 \
  -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -Dncnn_DIR="$PWD/build/ncnn-install/lib/cmake/ncnn"
cmake --build build/phone -j2
```

Select an authorized ADB device, then transfer the inference files. Replace the
serial placeholder. `push_models.py` checks the device identity and resumes
transfers using saved checksums; encoder TSV and the PPM input need explicit copies.

```sh
PHONE_SERIAL=replace-with-your-authorized-serial
PHONE_ID=$(adb -s "$PHONE_SERIAL" shell getprop ro.serialno | tr -d '\r')
python mobile/push_models.py --adb adb --serial "$PHONE_SERIAL" \
  --expected-phone "$PHONE_ID" --codec build/models-256/codec \
  --renderer build/models-256/renderer \
  --destination /data/local/tmp/aeic-phone/model256 --state build/transfer-state.json
adb -s "$PHONE_SERIAL" push build/phone/aeic_decode /data/local/tmp/aeic-phone/
adb -s "$PHONE_SERIAL" push build/models-256/codec/encoder.tsv /data/local/tmp/aeic-phone/model256/
adb -s "$PHONE_SERIAL" push build/models-256/codec/input.ppm /data/local/tmp/aeic-phone/
adb -s "$PHONE_SERIAL" shell chmod 755 /data/local/tmp/aeic-phone/aeic_decode
```

## Encode and decode on the phone

```sh
adb -s "$PHONE_SERIAL" shell /data/local/tmp/aeic-phone/aeic_decode \
  --encode --model-dir /data/local/tmp/aeic-phone/model256 \
  --input /data/local/tmp/aeic-phone/input.ppm \
  --bitstream-output /data/local/tmp/aeic-phone/phone.aeic --threads 2
adb -s "$PHONE_SERIAL" shell /data/local/tmp/aeic-phone/aeic_decode \
  --model-dir /data/local/tmp/aeic-phone/model256 \
  --bitstream /data/local/tmp/aeic-phone/phone.aeic \
  --output /data/local/tmp/aeic-phone/reconstructed.png --threads 2
adb -s "$PHONE_SERIAL" pull /data/local/tmp/aeic-phone/reconstructed.png .
```

Encoding success prints `ENCODE_COMPLETE`. A complete image decode must print
`DECODE_COMPLETE`; `--stop-after y_hat` checks entropy only and prints
`PARTIAL_TEST`. See [native/README.md](native/README.md) for exact integer
verification, float comparisons, stage diagnostics, and host native tests.

Codec and encoder graphs, weights, and entropy tables are byte-identical across
the verified 256 and 512 exports. Renderer graph parameters depend on size.
Use the actual resolution's renderer TSV/parameters and `--model-size 512` when
retaining 256 metadata. Check renderer weight hashes between revised exports
before reusing their binary files across resolutions.
Keep TorchScript, CPU reference arrays, pretrained source checkpoints, and
transfer authentication tokens on the host; they are not phone inference assets.
