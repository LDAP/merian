#include "merian/vk/imgui/imgui_renderer.hpp"
#include "merian/vk/imgui/extension_imgui.hpp"

#include "imgui.frag.spv.h"
#include "imgui.vert.spv.h"

#include "merian/shader/entry_point.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/pipeline/pipeline_graphics_builder.hpp"
#include "merian/vk/pipeline/pipeline_layout.hpp"

#include <limits>
#include <stdexcept>

namespace merian {

namespace {

struct PushConstants {
    float scale[2];
    float translate[2];
};

} // namespace

ImGuiRenderer::ImGuiRenderer(const ContextHandle& context,
                             const ResourceAllocatorHandle& alloc,
                             const ImGuiContextHandle& imgui_ctx)
    : context(context), alloc(alloc), imgui_ctx(imgui_ctx) {
    if (!context->get_context_extension<ExtensionImGui>(/*null_ok=*/true)) {
        throw std::runtime_error("ImGuiRenderer requires the 'merian-imgui' context extension. "
                                 "Add it to your ContextCreateInfo::context_extensions.");
    }
    imgui_ctx->get_io().BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
}

ImGuiRenderer::~ImGuiRenderer() = default;

// Finds or creates a free slot in the texture registry, starting at index 1.
// Slot 0 is reserved (ImTextureID_Invalid == 0).
std::size_t ImGuiRenderer::alloc_texture_slot() {
    if (textures.empty())
        textures.push_back(nullptr); // reserve slot 0
    for (std::size_t i = 1; i < textures.size(); ++i) {
        if (!textures[i])
            return i;
    }
    textures.emplace_back(nullptr);
    return textures.size() - 1;
}

ImTextureID ImGuiRenderer::register_texture(const TextureHandle& texture) {
    const std::size_t slot = alloc_texture_slot();
    textures[slot] = texture;
    return static_cast<ImTextureID>(slot);
}

void ImGuiRenderer::unregister_texture(ImTextureID id) {
    const auto idx = static_cast<std::size_t>(id);
    if (idx == 0 || idx >= textures.size())
        return;
    textures[idx] = nullptr;
}

void ImGuiRenderer::init_pipeline(vk::Format color_format) {
    descriptor_layout =
        DescriptorSetLayoutBuilder().add_binding_combined_sampler().build_push_descriptor_layout(
            context);

    const vk::PushConstantRange pc_range{vk::ShaderStageFlagBits::eVertex, 0,
                                         sizeof(PushConstants)};

    pipeline_layout = std::make_shared<PipelineLayout>(
        context, std::vector<DescriptorSetLayoutHandle>{descriptor_layout},
        std::vector<vk::PushConstantRange>{pc_range});

    auto vert = EntryPoint::create(context, merian_imgui_vert_spv(), merian_imgui_vert_spv_size(),
                                   "main", vk::ShaderStageFlagBits::eVertex);
    auto frag = EntryPoint::create(context, merian_imgui_frag_spv(), merian_imgui_frag_spv_size(),
                                   "main", vk::ShaderStageFlagBits::eFragment);

    pipeline =
        GraphicsPipelineBuilder()
            .set_vertex_shader(vert)
            .set_fragment_shader(frag)
            .vertex_input_add_binding({0, sizeof(ImDrawVert), vk::VertexInputRate::eVertex})
            .vertex_input_add_attribute(
                {0, 0, vk::Format::eR32G32Sfloat, offsetof(ImDrawVert, pos)})
            .vertex_input_add_attribute({1, 0, vk::Format::eR32G32Sfloat, offsetof(ImDrawVert, uv)})
            .vertex_input_add_attribute(
                {2, 0, vk::Format::eR8G8B8A8Unorm, offsetof(ImDrawVert, col)})
            .rasterizer_cull_mode(vk::CullModeFlagBits::eNone)
            .blend_add_attachment(VK_TRUE, vk::BlendFactor::eSrcAlpha,
                                  vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd,
                                  vk::BlendFactor::eOne, vk::BlendFactor::eOneMinusSrcAlpha,
                                  vk::BlendOp::eAdd)
            .dyanmic_state_add(vk::DynamicState::eViewport)
            .dyanmic_state_add(vk::DynamicState::eScissor)
            .viewport_add(1.0f, 1.0f) // count=1, overridden by dynamic state
            .build_dynamic_rendering(pipeline_layout, color_format);

    current_format = color_format;
}

void ImGuiRenderer::upload_pending_textures(const CommandBufferHandle& cmd,
                                            ImVector<ImTextureData*>* tex_list) {
    if ((tex_list == nullptr) || tex_list->Size == 0)
        return;
    for (int i = 0; i < tex_list->Size; ++i) {
        ImTextureData* tex = (*tex_list)[i];
        if (tex->Status == ImTextureStatus_WantCreate ||
            tex->Status == ImTextureStatus_WantUpdates) {
            if (tex->Format != ImTextureFormat_RGBA32)
                throw std::runtime_error("ImGuiRenderer: only RGBA32 textures are supported");

            auto texture = alloc->create_texture_from_rgba8(
                cmd, reinterpret_cast<const uint32_t*>(tex->Pixels),
                static_cast<uint32_t>(tex->Width), static_cast<uint32_t>(tex->Height),
                vk::SamplerAddressMode::eRepeat, vk::Filter::eLinear, vk::Filter::eLinear, false,
                "imgui_atlas");
            cmd->barrier(texture->get_image()->barrier2(
                vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eTransferWrite,
                vk::AccessFlagBits2::eShaderSampledRead, vk::PipelineStageFlagBits2::eTransfer,
                vk::PipelineStageFlagBits2::eFragmentShader));

            if (tex->Status == ImTextureStatus_WantCreate) {
                const std::size_t slot = alloc_texture_slot();
                textures[slot] = texture;
                tex->SetTexID(static_cast<ImTextureID>(slot));
            } else {
                textures[static_cast<std::size_t>(tex->GetTexID())] = texture;
            }
            tex->SetStatus(ImTextureStatus_OK);
        } else if (tex->Status == ImTextureStatus_WantDestroy) {
            const auto slot = static_cast<std::size_t>(tex->GetTexID());
            if (slot < textures.size())
                textures[slot] = nullptr;
            tex->SetTexID(0);
            tex->BackendUserData = nullptr;
            tex->SetStatus(ImTextureStatus_Destroyed);
        }
    }
}

void ImGuiRenderer::ensure_buffers(const ImDrawData* draw_data) {
    const vk::DeviceSize vtx_size =
        static_cast<vk::DeviceSize>(draw_data->TotalVtxCount) * sizeof(ImDrawVert);
    const vk::DeviceSize idx_size =
        static_cast<vk::DeviceSize>(draw_data->TotalIdxCount) * sizeof(ImDrawIdx);
    alloc->ensure_buffer_size(vertex_buffer, vtx_size,
                              vk::BufferUsageFlagBits::eVertexBuffer |
                                  vk::BufferUsageFlagBits::eTransferDst,
                              "imgui_vertices");
    alloc->ensure_buffer_size(index_buffer, idx_size,
                              vk::BufferUsageFlagBits::eIndexBuffer |
                                  vk::BufferUsageFlagBits::eTransferDst,
                              "imgui_indices");
}

void ImGuiRenderer::render(const CommandBufferHandle& cmd,
                           const ImageViewHandle& image_view,
                           const ProfilerHandle& profiler) {
    MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "ImGui::render");

    ImDrawData* draw_data = nullptr;
    imgui_ctx->with_context([&] {
        ImGui::Render();
        draw_data = ImGui::GetDrawData();
    });

    if ((draw_data == nullptr) || draw_data->TotalVtxCount == 0)
        return;

    const ImageHandle image = image_view->get_image();
    const vk::Format format = image->get_format();

    if (!initialized || format != current_format) {
        init_pipeline(format);
        initialized = true;
    }

    {
        MERIAN_PROFILE_SCOPE(profiler, "ImGui::upload_textures");
        upload_pending_textures(cmd, draw_data->Textures);
    }

    ensure_buffers(draw_data);

    // Upload vertex and index data via staging (staging manager rotates internally)
    {
        MERIAN_PROFILE_SCOPE(profiler, "ImGui::upload_buffers");
        vk::DeviceSize vtx_offset = 0;
        vk::DeviceSize idx_offset = 0;
        for (int n = 0; n < draw_data->CmdListsCount; ++n) {
            const ImDrawList* list = draw_data->CmdLists[n];
            const vk::DeviceSize vtx_bytes =
                static_cast<vk::DeviceSize>(list->VtxBuffer.Size) * sizeof(ImDrawVert);
            const vk::DeviceSize idx_bytes =
                static_cast<vk::DeviceSize>(list->IdxBuffer.Size) * sizeof(ImDrawIdx);
            alloc->get_staging()->cmd_to_device(cmd, vertex_buffer, list->VtxBuffer.Data,
                                                vtx_offset, vtx_bytes);
            alloc->get_staging()->cmd_to_device(cmd, index_buffer, list->IdxBuffer.Data, idx_offset,
                                                idx_bytes);
            vtx_offset += vtx_bytes;
            idx_offset += idx_bytes;
        }
    }

    // Barrier: transfer writes must complete before vertex/index reads in the draw
    cmd->barrier(vertex_buffer->buffer_barrier2(
        vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eVertexAttributeInput,
        vk::AccessFlagBits2::eTransferWrite, vk::AccessFlagBits2::eVertexAttributeRead));
    cmd->barrier(index_buffer->buffer_barrier2(
        vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eIndexInput,
        vk::AccessFlagBits2::eTransferWrite, vk::AccessFlagBits2::eIndexRead));

    // Transition swapchain image: current → eColorAttachmentOptimal (preserve contents for eLoad)
    cmd->barrier(image->barrier2(vk::ImageLayout::eColorAttachmentOptimal));

    // Begin dynamic rendering
    const vk::RenderingAttachmentInfo color_attachment{*image_view,
                                                       vk::ImageLayout::eColorAttachmentOptimal,
                                                       vk::ResolveModeFlagBits::eNone,
                                                       {},
                                                       {},
                                                       vk::AttachmentLoadOp::eLoad,
                                                       vk::AttachmentStoreOp::eStore,
                                                       {}};
    // DisplaySize is in logical pixels; scale to physical pixels for the render target.
    const float fb_width = draw_data->DisplaySize.x * draw_data->FramebufferScale.x;
    const float fb_height = draw_data->DisplaySize.y * draw_data->FramebufferScale.y;
    const vk::Extent2D fb_extent{static_cast<uint32_t>(fb_width), static_cast<uint32_t>(fb_height)};
    cmd->begin_rendering(
        vk::RenderingInfo{{}, vk::Rect2D{{0, 0}, fb_extent}, 1, 0, color_attachment});

    cmd->bind(pipeline);
    cmd->set_viewport(vk::Viewport{0, 0, fb_width, fb_height, 0.0f, 1.0f});

    PushConstants pc;
    pc.scale[0] = 2.0f / draw_data->DisplaySize.x;
    pc.scale[1] = 2.0f / draw_data->DisplaySize.y;
    pc.translate[0] = -1.0f - (draw_data->DisplayPos.x * pc.scale[0]);
    pc.translate[1] = -1.0f - (draw_data->DisplayPos.y * pc.scale[1]);
    cmd->push_constant(pipeline, pc);

    cmd->bind_vertex_buffer(vertex_buffer);
    cmd->bind_index_buffer(index_buffer, sizeof(ImDrawIdx) == 2 ? vk::IndexType::eUint16
                                                                : vk::IndexType::eUint32);

    const ImVec2 clip_off = draw_data->DisplayPos;
    const ImVec2 clip_scale = draw_data->FramebufferScale;
    uint32_t vtx_offset = 0;
    uint32_t idx_offset = 0;
    std::size_t last_tex_id = std::numeric_limits<std::size_t>::max();
    for (int n = 0; n < draw_data->CmdListsCount; ++n) {
        const ImDrawList* list = draw_data->CmdLists[n];
        for (int cmd_i = 0; cmd_i < list->CmdBuffer.Size; ++cmd_i) {
            const ImDrawCmd& draw_cmd = list->CmdBuffer[cmd_i];
            if (draw_cmd.UserCallback != nullptr) {
                draw_cmd.UserCallback(list, &draw_cmd);
                continue;
            }

            const auto tex_id = static_cast<std::size_t>(draw_cmd.GetTexID());
            if (tex_id != last_tex_id && tex_id < textures.size() && textures[tex_id]) {
                cmd->push_descriptor_set(pipeline, textures[tex_id]);
                last_tex_id = tex_id;
            }

            const float cx0 = (draw_cmd.ClipRect.x - clip_off.x) * clip_scale.x;
            const float cy0 = (draw_cmd.ClipRect.y - clip_off.y) * clip_scale.y;
            const float cx1 = (draw_cmd.ClipRect.z - clip_off.x) * clip_scale.x;
            const float cy1 = (draw_cmd.ClipRect.w - clip_off.y) * clip_scale.y;

            if (cx1 <= cx0 || cy1 <= cy0)
                continue;

            cmd->set_scissor(
                vk::Rect2D{{static_cast<int32_t>(cx0), static_cast<int32_t>(cy0)},
                           {static_cast<uint32_t>(cx1 - cx0), static_cast<uint32_t>(cy1 - cy0)}});

            cmd->draw_indexed(draw_cmd.ElemCount, 1, draw_cmd.IdxOffset + idx_offset,
                              static_cast<int32_t>(draw_cmd.VtxOffset + vtx_offset));
        }
        vtx_offset += static_cast<uint32_t>(list->VtxBuffer.Size);
        idx_offset += static_cast<uint32_t>(list->IdxBuffer.Size);
    }

    cmd->end_rendering();
}

} // namespace merian
