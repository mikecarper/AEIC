// Temporary authenticated receiver for files in the AEIC experiment directory.
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cerrno>
#include <string>
#include <vector>

static bool exact(int fd, void* data, size_t size) {
    char* p = static_cast<char*>(data);
    while (size) {
        ssize_t n = read(fd, p, size);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        p += n; size -= size_t(n);
    }
    return true;
}
static uint64_t little(const unsigned char* p, int bytes) {
    uint64_t value = 0;
    for (int i = 0; i < bytes; ++i) value |= uint64_t(p[i]) << (8 * i);
    return value;
}
static bool filename(const std::string& name) {
    if (name.empty() || name.size() > 200 || name.find("..") != std::string::npos) return false;
    for (unsigned char c : name)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-')) return false;
    return true;
}
int main(int argc, char** argv) {
    if (argc == 3 && !strcmp(argv[1], "--hash")) {
        FILE* file = fopen(argv[2], "rb");
        if (!file) return 1;
        unsigned char buffer[65536];
        uint64_t hash = UINT64_C(14695981039346656037);
        size_t count;
        while ((count = fread(buffer, 1, sizeof(buffer), file)))
            for (size_t i = 0; i < count; ++i) { hash ^= buffer[i]; hash *= UINT64_C(1099511628211); }
        if (ferror(file)) { fclose(file); return 1; }
        fclose(file);
        printf("%llu\n", static_cast<unsigned long long>(hash)); return 0;
    }
    if (argc != 4) { fprintf(stderr, "usage: receive_models DIRECTORY TOKEN_FILE PORT\n"); return 2; }
    std::string directory = argv[1];
    if (directory != "/data/local/tmp/aeic-phone/codec256") return 2;
    FILE* secret = fopen(argv[2], "rb");
    char token[32];
    if (!secret || fread(token, 1, 32, secret) != 32) return 2;
    fclose(secret);
    int port = atoi(argv[3]);
    if (port < 1024 || port > 65535) return 2;
    int server = socket(AF_INET, SOCK_STREAM, 0), reuse = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in address{}; address.sin_family = AF_INET;
    address.sin_port = htons(uint16_t(port)); address.sin_addr.s_addr = htonl(INADDR_ANY);
    if (server < 0 || bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) || listen(server, 1)) {
        perror("listen"); return 1;
    }
    printf("RECEIVER_READY port=%d\n", port); fflush(stdout);
    int client = -1;
    for (int tries = 0; tries < 10; ++tries) {
        client = accept(server, nullptr, nullptr);
        if (client < 0) return 1;
        timeval timeout{120, 0};
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        char greeting[40];
        if (exact(client, greeting, sizeof(greeting)) && !memcmp(greeting, "AEICPUT1", 8) &&
            !memcmp(greeting + 8, token, 32)) break;
        close(client); client = -1;
    }
    close(server);
    if (client < 0 || write(client, "A", 1) != 1) return 1;
    std::vector<unsigned char> buffer(65536);
    uint64_t total = 0;
    while (true) {
        unsigned char header[18];
        if (!exact(client, header, sizeof(header))) return 1;
        size_t length = size_t(little(header, 2));
        uint64_t size = little(header + 2, 8), expected = little(header + 10, 8);
        if (!length && !size) break;
        if (length > 200 || size > (128u << 20)) return 1;
        std::string name(length, ' ');
        if (!exact(client, &name[0], length) || !filename(name)) return 1;
        std::string final = directory + "/" + name, temporary = final + ".incoming";
        int file = open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600);
        if (file < 0) return 1;
        uint64_t hash = UINT64_C(14695981039346656037), remaining = size;
        bool okay = true;
        while (remaining) {
            size_t count = size_t(remaining < buffer.size() ? remaining : buffer.size());
            if (!exact(client, buffer.data(), count)) { okay = false; break; }
            size_t offset = 0;
            while (offset < count) {
                ssize_t written = write(file, buffer.data() + offset, count - offset);
                if (written < 0 && errno == EINTR) continue;
                if (written <= 0) { okay = false; break; }
                offset += size_t(written);
            }
            if (!okay) break;
            for (size_t i = 0; i < count; ++i) { hash ^= buffer[i]; hash *= UINT64_C(1099511628211); }
            remaining -= count;
        }
        if (close(file)) okay = false;
        if (!okay || hash != expected || rename(temporary.c_str(), final.c_str())) {
            unlink(temporary.c_str()); return 1;
        }
        total += size;
        printf("RECEIVED %s bytes=%llu\n", name.c_str(), static_cast<unsigned long long>(size)); fflush(stdout);
        if (write(client, "A", 1) != 1) return 1;
    }
    close(client);
    printf("TRANSFER_COMPLETE received_bytes=%llu\n", static_cast<unsigned long long>(total));
    return 0;
}
