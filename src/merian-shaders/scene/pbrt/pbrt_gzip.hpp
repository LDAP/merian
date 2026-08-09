#pragma once

#include <filesystem>
#include <string>

namespace merian::pbrt {

// Reads gzip-compressed or plain files transparently (zlib passes uncompressed data through).
std::string gunzip_file(const std::filesystem::path& path);

} // namespace merian::pbrt
