#pragma once

#include "merian-shaders/shading/materials/material_system.hpp"
#include "merian/utils/camera/camera.hpp"
#include "merian/utils/vector_matrix.hpp"

#include <cstdint>
#include <string>

namespace merian {

// Abstract class for scenes
class Scene {

  public:
    using NodeID = uint32_t;
    static constexpr NodeID NODE_ID_INVALID = uint32_t(-1);

    /**
     * This class describes a node in a scene graph.
     */
    class Node {
        Node() : parent(NODE_ID_INVALID), name(), transform(identity()) {}

        const NodeID& get_parent() const {
            return parent;
        }

        const std::string& get_name() const {
            return name;
        }

        const float4x4& get_transform() const {
            return transform;
        }

      private:
        NodeID parent;
        std::string name;
        float4x4 transform;
    };

  protected:
    std::vector<Node> scene_graph;

    std::vector<CameraHandle> cameras;
    uint32_t active_camera = 0;
    MaterialSystemHandle material_system;
};

} // namespace merian
