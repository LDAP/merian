#pragma once

#include "merian/shader/shader_cursor.hpp"
#include "merian/shader/slang_composition.hpp"

#include <memory>
#include <string>
#include <vector>

namespace merian {

class IrradianceCache {
  public:
    virtual ~IrradianceCache() = default;

    virtual SlangCompositionHandle get_composition() const = 0;

    virtual std::vector<std::string> get_slang_imports() const = 0;

    virtual std::string get_type_name() const = 0;

    virtual void write_to(ShaderCursor cursor) = 0;
};

using IrradianceCacheHandle = std::shared_ptr<IrradianceCache>;

class NullIrradianceCache : public IrradianceCache {
  public:
    SlangCompositionHandle get_composition() const override {
        const SlangCompositionHandle composition = SlangComposition::create();
        composition->add_module_from_path(MODULE_PATH);
        return composition;
    }

    std::vector<std::string> get_slang_imports() const override {
        return {slang_import_spelling(MODULE_PATH)};
    }

    std::string get_type_name() const override {
        return "merian::NullIrradianceCache";
    }

    void write_to([[maybe_unused]] ShaderCursor cursor) override {}

  private:
    static constexpr const char* MODULE_PATH = "merian-shaders/light-cache/irradiance-cache.slang";
};

} // namespace merian
