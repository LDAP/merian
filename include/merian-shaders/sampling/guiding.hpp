#pragma once

#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/utils/properties.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <memory>
#include <string>
#include <vector>

namespace merian {

/**
 * @brief Host side of a directional guiding method.
 *
 * Mirrors the GuidingModel interface in merian-shaders/sampling/guiding.slang: get_type_name()
 * names the type a renderer aliases into its guiding slot, the imports and the composition make
 * that name resolve, and write_to() fills the slot in the renderer's parameter block.
 */
class GuidingModel {
  public:
    virtual ~GuidingModel() = default;

    // Device resources, once the node holding the method has them. Properties are loaded before
    // this, so the constructor must not need a device.
    virtual void initialize(const ContextHandle& context,
                            const ResourceAllocatorHandle& allocator) = 0;

    virtual SlangCompositionHandle get_composition() const = 0;

    // Modules the renderer's slot module has to import for get_type_name() to resolve.
    virtual std::vector<std::string> get_slang_imports() const = 0;

    virtual std::string get_type_name() const = 0;

    // Binds the method for the frame about to be rendered.
    virtual void write_to(ShaderCursor cursor) = 0;

    virtual void reset(const CommandBufferHandle& cmd) = 0;

    // Returns true where the renderer has to recompile.
    virtual bool properties(Properties& props) = 0;
};

using GuidingModelHandle = std::shared_ptr<GuidingModel>;

class NullGuidingModel : public GuidingModel {
  public:
    void initialize([[maybe_unused]] const ContextHandle& context,
                    [[maybe_unused]] const ResourceAllocatorHandle& allocator) override {}

    SlangCompositionHandle get_composition() const override {
        return SlangComposition::create();
    }

    std::vector<std::string> get_slang_imports() const override {
        return {};
    }

    std::string get_type_name() const override {
        return "merian::NullGuidingModel";
    }

    void write_to([[maybe_unused]] ShaderCursor cursor) override {}

    void reset([[maybe_unused]] const CommandBufferHandle& cmd) override {}

    bool properties([[maybe_unused]] Properties& props) override {
        return false;
    }
};

} // namespace merian
