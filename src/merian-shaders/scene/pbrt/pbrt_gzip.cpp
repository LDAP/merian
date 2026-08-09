#include "pbrt_gzip.hpp"

#include <stdexcept>
#include <zlib.h>

namespace merian::pbrt {

std::string gunzip_file(const std::filesystem::path& path) {
    gzFile file = gzopen(path.string().c_str(), "rb");
    if (file == nullptr) {
        throw std::runtime_error("cannot open " + path.string());
    }
    gzbuffer(file, 256 * 1024);

    std::string result;
    char buffer[64 * 1024];
    int read_count;
    while ((read_count = gzread(file, buffer, sizeof(buffer))) > 0) {
        result.append(buffer, static_cast<size_t>(read_count));
    }
    if (read_count < 0) {
        int err_code;
        const char* err = gzerror(file, &err_code);
        const std::string message = err != nullptr ? err : "gzread failed";
        gzclose(file);
        throw std::runtime_error(path.string() + ": " + message);
    }
    gzclose(file);
    return result;
}

} // namespace merian::pbrt
