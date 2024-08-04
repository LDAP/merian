#include "merian/io/file_loader.hpp"
#include "tiny_obj_loader.h"
#include <optional>

namespace merian {

// Convenience method that reads a .obj file using tinyobjloader and throws if the file cound not be
// found or the input file is not valid.
tinyobj::ObjReader read_obj(std::string filename,
                                   std::optional<FileLoader> loader = std::nullopt);

} // namespace merian
