#pragma once

#include <string>
#include <unistd.h>

#ifdef _WIN32
#include <cstdio>
#endif

namespace merian {

inline std::string temporary_file() {
#ifdef _WIN32
    return std::tmpnam(nullptr);
#else
    // std::tmpnam is deprecated
    const char* tem_dir_name = std::getenv("TMPDIR");
    std::string tmp_file_name_template = (tem_dir_name != nullptr) ? tem_dir_name : "";
#ifdef P_tmpdir
    if (tmp_file_name_template.empty()) {
        tmp_file_name_template = P_tmpdir;
    }
#endif
    if (tmp_file_name_template.empty()) {
        tmp_file_name_template = "/tmp";
    }

    tmp_file_name_template += "/merianXXXXXX";
    const int fd = mkstemp(const_cast<char*>(tmp_file_name_template.c_str()));
    if (fd > 0) {
        // immediately close again...
        close(fd);
    }

    return tmp_file_name_template;
#endif
}

} // namespace merian
