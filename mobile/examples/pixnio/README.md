# Actual Android 5.1 phone round trip

This is a real Pixnio photograph, scaled with Lanczos and center-cropped to
128 by 128 pixels before encoding. The phone generated the 137-byte
`phone-encoded128.aeic` file and decoded that same file into
`phone-decoded128.png`. The encoded file is byte-identical to the original
CPU codec. Encoding took 2.172 seconds; complete decoding took 114.477 seconds
and peaked at 242,204 KiB process RSS on the LG L51AL / Android 5.1.1.

| Scaled original | Phone reconstruction |
|---|---|
| ![Original](original128.png) | ![Decoded on phone](phone-decoded128.png) |

The small transmitted file relies on the separately installed FT2 learned
models, which occupy about 2 GB. It is lossy; the PNG is a viewing copy, not
the transmitted file. See the [mobile instructions](../../README.md).

`phone-recovered128.png` is a separate best-effort test. Four bytes were removed
from the compressed file, one renderer layer was missing, and an entropy
reference index was changed. Strict decoding rejected the file; opt-in recovery
produced this explicitly degraded PNG in 137.142 seconds with six recovery
warnings. Recovery cannot restore information that was lost.

![Degraded best-effort phone reconstruction](phone-recovered128.png)

Source: [Pixnio river / forest photograph](https://pixnio.com/nature-landscapes/river/landscape-water-nature-tree-river-forest-mountain),
licensed [CC0](https://creativecommons.org/publicdomain/zero/1.0/).
