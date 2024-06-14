#pragma once

#include "merian-nodes/graph/resource.hpp"

#include "merian/vk/memory/resource_allocations.hpp"

namespace merian_nodes {

class TextureArrayResource : public GraphResource {
    friend class TextureArrayOut;
    friend class TextureArrayIn;

  public:
    TextureArrayResource(const uint32_t array_size, const uint32_t ring_size)
        : textures(array_size) {
        in_flight_textures.assign(ring_size, std::vector<merian::TextureHandle>(array_size));
    }

    void set(const uint32_t index, const merian::TextureHandle& tex) {
        textures[index] = tex;
        current_updates.push_back(index);
    }

    const merian::TextureHandle& get(const uint32_t index) const {
        return textures[index];
    }

  private:
    // the updates to "textures" are recorded here.
    std::vector<uint32_t> current_updates;
    // then flushed to here to wait for the graph to apply descriptor updates.
    std::vector<uint32_t> pending_updates;

    std::vector<merian::TextureHandle> textures;
    // on_post_process copy here to keep alive
    std::vector<std::vector<merian::TextureHandle>> in_flight_textures;
};

} // namespace merian_nodes
