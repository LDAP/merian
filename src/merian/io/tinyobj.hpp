#include "merian/io/file_loader.hpp"
#include "tiny_obj_loader.h"
#include <optional>

namespace merian {

// Convenience method that reads a .obj file using tinyobjloader and throws if the file cound not be
// found or the input file is not valid.
inline tinyobj::ObjReader read_obj(std::string filename,
                                   std::optional<FileLoader> loader = std::nullopt) {
    tinyobj::ObjReader reader;
    std::string full_path;

    if (loader.has_value()) {
        auto opt_path = loader.value().find_file(filename);
        if (!opt_path) {
            throw std::runtime_error{"file not found using loader"};
        }
        full_path = loader.value().find_file(filename).value();
    } else {
        full_path = filename;
    }
    reader.ParseFromFile(full_path);

    if (!reader.Valid()) {
        throw std::runtime_error{fmt::format("tinyobjloader: file {} not valid. {}{}", full_path,
                                             reader.Warning(), reader.Error())};
    }

    SPDLOG_DEBUG("read file {}, number vertices: {}, number materials: {}, number shapes: {},",
                 full_path, reader.GetAttrib().vertices.size(), reader.GetMaterials().size(),
                 reader.GetShapes().size());

    return reader;
}

} // namespace merian
