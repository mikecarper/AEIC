#include "png_writer.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <zlib.h>

namespace {
uint32_t be32(const uint8_t* bytes) {
    return uint32_t(bytes[0]) << 24 | uint32_t(bytes[1]) << 16 |
           uint32_t(bytes[2]) << 8 | uint32_t(bytes[3]);
}
void expect(bool value, const char* description) {
    if (!value) throw std::runtime_error(description);
}
}
int main() {
    try {
        std::vector<uint8_t> pixels{255,0,0, 0,255,0, 0,0,255, 255,255,255};
        auto png = aeic::make_png(2, 2, pixels);
        const uint8_t signature[] = {137,80,78,71,13,10,26,10};
        expect(png.size() > 8 && !std::memcmp(png.data(), signature, 8), "PNG signature");
        size_t cursor = 8; int count = 0;
        std::vector<uint8_t> compressed;
        while (cursor < png.size()) {
            expect(png.size() - cursor >= 12, "PNG chunk framing");
            uint32_t length = be32(png.data() + cursor);
            expect(length <= png.size() - cursor - 12, "PNG chunk length");
            const uint8_t* type = png.data() + cursor + 4;
            const uint8_t* data = type + 4;
            uLong crc = crc32(0L, Z_NULL, 0);
            crc = crc32(crc, type, length + 4);
            expect(static_cast<uint32_t>(crc) == be32(data + length), "PNG chunk CRC32");
            if (!std::memcmp(type, "IHDR", 4)) {
                expect(length == 13 && be32(data) == 2 && be32(data + 4) == 2 && data[8] == 8 && data[9] == 2, "RGB8 IHDR");
            } else if (!std::memcmp(type, "IDAT", 4)) compressed.insert(compressed.end(), data, data + length);
            else expect(!std::memcmp(type, "IEND", 4) && !length, "IEND chunk");
            cursor += length + 12; ++count;
        }
        expect(count == 3 && cursor == png.size(), "PNG three complete chunks");
        std::vector<uint8_t> decoded(14); uLongf length = decoded.size();
        expect(uncompress(decoded.data(), &length, compressed.data(), compressed.size()) == Z_OK && length == 14, "PNG zlib scanlines");
        expect(decoded[0] == 0 && decoded[7] == 0 &&
               !std::memcmp(decoded.data() + 1, pixels.data(), 6) &&
               !std::memcmp(decoded.data() + 8, pixels.data() + 6, 6), "PNG pixels roundtrip exactly");
        std::printf("PASS: native RGB8 PNG signature, dimensions, chunk CRC32, zlib scanlines, exact pixels\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1;
    }
}
