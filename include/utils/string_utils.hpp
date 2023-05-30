#include <cstdint>
#include <string>
#include <sstream>

namespace merian {

    inline std::string format_size(uint64_t size) {
        std::ostringstream oss;
        if (size < 1024) {
            oss << size << " B";
        } else if (size < 1024 * 1024) {
            oss << size / 1024.f << " KB";
        } else if (size < 1024 * 1024 * 1024) {
            oss << size / (1024.0f * 1024.0f) << " MB";
        } else {
            oss << size / (1024.0f * 1024.0f * 1024.0f) << " GB";
        }
        return oss.str();
    }

}
