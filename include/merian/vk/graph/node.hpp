#pragma once

#include "merian/vk/descriptors/descriptor_set.hpp"
#include "merian/vk/graph/node_io.hpp"
#include "vulkan/vulkan.hpp"

#include <memory>

namespace merian {

class Graph;

class Node : public std::enable_shared_from_this<Node> {
  public:
    static inline NodeOutputDescriptorImage FEEDBACK_OUTPUT_IMAGE{{}, {}, {}, {}, {}};
    static inline NodeOutputDescriptorBuffer FEEDBACK_OUTPUT_BUFFER{{}, {}, {}, {}};

    virtual std::string name() = 0;

    // Declare the inputs that you require
    virtual std::tuple<std::vector<NodeInputDescriptorImage>,
                       std::vector<NodeInputDescriptorBuffer>>
    describe_inputs() = 0;

    // Declare your outputs. Based on the output descriptors that were connected to your inputs.
    // You can check format and such here and fail if they are not compatible.
    // This may be called with different parameters when the graph is rebuilding.
    // Note: You do NOT get valid descriptors for delayed images and buffers, since those are
    // instanciated later, instead you get FEEDBACK_OUTPUT_IMAGE and FEEDBACK_OUTPUT_BUFFER
    // respectively.
    virtual std::tuple<std::vector<NodeOutputDescriptorImage>,
                       std::vector<NodeOutputDescriptorBuffer>>
    describe_outputs(const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
                     const std::vector<NodeOutputDescriptorBuffer>& connected_buffer_outputs) = 0;

    // Called when the graph is build or rebuild. You get your inputs and outputs for each set_index
    // (see cmd_process), use these to create your descriptor sets and such. You can also make and
    // such uploads here. You should only write to output images that were declared as 'persistent',
    // these are also the same in each set.
    // No not access or modify input images.
    virtual void cmd_build(const vk::CommandBuffer& cmd,
                           const std::vector<std::vector<ImageHandle>>& image_inputs,
                           const std::vector<std::vector<BufferHandle>>& buffer_inputs,
                           const std::vector<std::vector<ImageHandle>>& image_outputs,
                           const std::vector<std::vector<BufferHandle>>& buffer_outputs) = 0;

    // This is called once per iteration.
    // You do not need to insert barriers for node inputs and outputs.
    // Use the descriptor set according to set_index.
    // If you need to perform layout transistions use the barrier() methods of the images.
    virtual void cmd_process(const vk::CommandBuffer& cmd,
                             const uint64_t iteration,
                             const uint32_t set_index,
                             const std::vector<ImageHandle>& image_inputs,
                             const std::vector<BufferHandle>& buffer_inputs,
                             const std::vector<ImageHandle>& image_outputs,
                             const std::vector<BufferHandle>& buffer_outputs) = 0;
};

using NodeHandle = std::shared_ptr<Node>;

} // namespace merian
