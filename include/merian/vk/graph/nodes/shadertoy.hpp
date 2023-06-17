#pragma once

#include "glm/ext/vector_float2.hpp"
#include "merian/io/file_loader.hpp"
#include "merian/utils/stopwatch.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/descriptors/descriptor_set_update.hpp"
#include "merian/vk/graph/node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian {

class ShadertoyNode : public merian::Node {
  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

  public:
    struct PushConstant {
        glm::vec2 iResolution{};
        float iTime{};
        float iTimeDelta{};
        float iFrame{};
    };

  public:
    ShadertoyNode(const SharedContext context,
                  const ResourceAllocatorHandle alloc,
                  std::string path,
                  FileLoader loader,
                  uint32_t width = 1920,
                  uint32_t height = 1080)
        : context(context), alloc(alloc), width(width), height(height) {
        auto builder = DescriptorSetLayoutBuilder().add_binding_storage_image(); // result
        layout = builder.build_layout(context);

        auto shader = std::make_shared<ShaderModule>(context, path, loader);
        auto pipe_layout = PipelineLayoutBuilder(context)
                               .add_descriptor_set_layout(layout)
                               .add_push_constant<PushConstant>(vk::ShaderStageFlagBits::eCompute)
                               .build_pipeline_layout();
        auto spec_builder = SpecializationInfoBuilder();
        spec_builder.add_entry(local_size_x, local_size_y);
        auto spec_info = spec_builder.build();
        pipe = std::make_shared<ComputePipeline>(pipe_layout, shader, spec_info);

        sw.reset();
    }

    void set_resolution(uint32_t width = 1920, uint32_t height = 1080) {
        this->width = width;
        this->height = height;
    }

    std::string name() override {
        return "ShadertoyNode";
    }

    virtual std::tuple<std::vector<merian::NodeInputDescriptorImage>,
                       std::vector<merian::NodeInputDescriptorBuffer>>
    describe_inputs() override {
        return {};
    }

    virtual std::tuple<std::vector<merian::NodeOutputDescriptorImage>,
                       std::vector<merian::NodeOutputDescriptorBuffer>>
    describe_outputs(const std::vector<merian::NodeOutputDescriptorImage>&,
                     const std::vector<merian::NodeOutputDescriptorBuffer>&) override {

        vk::ImageCreateInfo create_image{{},
                                         vk::ImageType::e2D,
                                         vk::Format::eR8G8B8A8Unorm,
                                         {width, height, 1},
                                         1,
                                         1,
                                         vk::SampleCountFlagBits::e1,
                                         vk::ImageTiling::eOptimal,
                                         vk::ImageUsageFlagBits::eStorage,
                                         vk::SharingMode::eExclusive,
                                         {},
                                         {},
                                         vk::ImageLayout::eUndefined};
        return {
            {
                merian::NodeOutputDescriptorImage{"result", vk::AccessFlagBits2::eShaderWrite,
                                                  vk::PipelineStageFlagBits2::eComputeShader,
                                                  create_image, vk::ImageLayout::eGeneral, false},
            },
            {},
        };
    }

    virtual void cmd_build(const vk::CommandBuffer&,
                           const std::vector<std::vector<merian::ImageHandle>>&,
                           const std::vector<std::vector<merian::BufferHandle>>&,
                           const std::vector<std::vector<merian::ImageHandle>>& image_outputs,
                           const std::vector<std::vector<merian::BufferHandle>>&) override {
        sets.clear();
        textures.clear();

        uint32_t num_sets = image_outputs.size();

        pool = std::make_shared<merian::DescriptorPool>(layout, num_sets);
        vk::ImageViewCreateInfo create_image_view{{},
                                                  VK_NULL_HANDLE,
                                                  vk::ImageViewType::e2D,
                                                  vk::Format::eR8G8B8A8Unorm,
                                                  {},
                                                  {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};

        for (uint32_t i = 0; i < num_sets; i++) {
            auto set = std::make_shared<merian::DescriptorSet>(pool);
            create_image_view.image = *image_outputs[i][0];
            auto tex = alloc->createTexture(image_outputs[i][0], create_image_view);
            DescriptorSetUpdate(set).write_descriptor_texture(0, tex).update(context);
            sets.push_back(set);
            textures.push_back(tex);
        }
        constant.iResolution = glm::vec2(width, height);
    }

    virtual void cmd_process(const vk::CommandBuffer& cmd,
                             const uint64_t iteration,
                             const uint32_t set_index,
                             const std::vector<merian::ImageHandle>&,
                             const std::vector<merian::BufferHandle>&,
                             const std::vector<merian::ImageHandle>&,
                             const std::vector<merian::BufferHandle>&) override {
        float new_time = sw.seconds();
        constant.iTimeDelta = new_time - constant.iTime;
        constant.iTime = new_time;
        constant.iFrame = iteration;

        pipe->bind(cmd);
        pipe->bind_descriptor_set(cmd, sets[set_index]);
        pipe->push_constant<PushConstant>(cmd, constant);
        cmd.dispatch((width + local_size_x - 1) / local_size_x,
                     (height + local_size_y - 1) / local_size_y, 1);
    }

  private:
    const SharedContext context;
    const ResourceAllocatorHandle alloc;

    uint32_t width;
    uint32_t height;

    DescriptorSetLayoutHandle layout;
    DescriptorPoolHandle pool;
    std::vector<DescriptorSetHandle> sets;
    std::vector<TextureHandle> textures;
    PipelineHandle pipe;

    PushConstant constant;
    Stopwatch sw;
};

} // namespace merian
