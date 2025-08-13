/*
 * Parts of this code were adapted NVCore which is licensed under:
 *
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include "merian/vk/sampler/sampler_pool.hpp"
#include <spdlog/spdlog.h>

namespace merian {
//////////////////////////////////////////////////////////////////////////

SamplerPool::~SamplerPool() {
    SPDLOG_DEBUG("destroy sampler pool ({})", fmt::ptr(this));
}

SamplerHandle SamplerPool::acquire_sampler(const vk::SamplerCreateInfo& createInfo) {
    SamplerState state{};
    state.createInfo = createInfo;

    const Chain* ext = (const Chain*)createInfo.pNext;
    while (ext) {
        switch (ext->sType) {
        case VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO:
            state.reduction = *(const VkSamplerReductionModeCreateInfo*)ext;
            break;
        case VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO:
            state.ycbr = *(const VkSamplerYcbcrConversionCreateInfo*)ext;
            break;
        default:
            std::runtime_error{"unsupported sampler create"};
        }
        ext = ext->pNext;
    }

    // always remove pointers for comparison lookup
    state.createInfo.pNext = nullptr;
    state.reduction.pNext = nullptr;
    state.ycbr.pNext = nullptr;

    auto it = state_map.find(state);

    if (it == state_map.end()) {
        SamplerHandle sampler = std::make_shared<Sampler>(context, createInfo);
        state_map[state] = sampler;
        return sampler;
    } else {
        if ((*it).second.expired()) {
            // recreate sampler
            auto sampler = std::make_shared<Sampler>(context, createInfo);
            state_map[state] = sampler;
            return sampler;

        } else {
            return (*it).second.lock();
        }
    }
}

SamplerHandle SamplerPool::acquire_sampler(const vk::Filter mag_filter,
                                                       const vk::Filter min_filter,
                                                       const vk::SamplerAddressMode address_mode_u,
                                                       const vk::SamplerAddressMode address_mode_v,
                                                       const vk::SamplerAddressMode address_mode_w,
                                                       const vk::SamplerMipmapMode mipmap_mode,
                                                       const bool anisotropy,
                                                       const vk::BorderColor border_color) {
    const vk::SamplerCreateInfo info{
        {},
        mag_filter,
        min_filter,
        mipmap_mode,
        address_mode_u,
        address_mode_v,
        address_mode_w,
        {},
        anisotropy ? VK_TRUE : VK_FALSE,
        context->physical_device.get_physical_device_limits().maxSamplerAnisotropy,
        VK_FALSE,
        {},
        0.0f,
        VK_LOD_CLAMP_NONE,
        border_color,
        VK_FALSE};
    return acquire_sampler(info);
}

SamplerHandle SamplerPool::for_filter_and_address_mode(const vk::Filter mag_filter,
                                                       const vk::Filter min_filter,
                                                       const vk::SamplerAddressMode address_mode,
                                                       const vk::SamplerMipmapMode mipmap_mode,
                                                       const bool anisotropy,
                                                       const vk::BorderColor border_color) {
    return acquire_sampler(mag_filter, min_filter, address_mode, address_mode,
                                       address_mode, mipmap_mode, anisotropy, border_color);
}

SamplerHandle SamplerPool::linear_repeat() {
    return for_filter_and_address_mode(vk::Filter::eLinear, vk::Filter::eLinear,
                                       vk::SamplerAddressMode::eRepeat,
                                       vk::SamplerMipmapMode::eLinear);
}

SamplerHandle SamplerPool::linear_mirrored_repeat() {
    return for_filter_and_address_mode(vk::Filter::eLinear, vk::Filter::eLinear,
                                       vk::SamplerAddressMode::eMirroredRepeat,
                                       vk::SamplerMipmapMode::eLinear);
}

SamplerHandle SamplerPool::linear_clamp_to_edge() {
    return for_filter_and_address_mode(vk::Filter::eLinear, vk::Filter::eLinear,
                                       vk::SamplerAddressMode::eClampToEdge,
                                       vk::SamplerMipmapMode::eLinear);
}

SamplerHandle SamplerPool::linear_clamp_to_border(const vk::BorderColor border_color) {
    return for_filter_and_address_mode(vk::Filter::eLinear, vk::Filter::eLinear,
                                       vk::SamplerAddressMode::eClampToBorder,
                                       vk::SamplerMipmapMode::eLinear, true, border_color);
}

SamplerHandle SamplerPool::nearest_mirrored_repeat() {
    return for_filter_and_address_mode(vk::Filter::eNearest, vk::Filter::eNearest,
                                       vk::SamplerAddressMode::eMirroredRepeat,
                                       vk::SamplerMipmapMode::eNearest);
}

SamplerHandle SamplerPool::nearest_repeat() {
    return for_filter_and_address_mode(vk::Filter::eNearest, vk::Filter::eNearest,
                                       vk::SamplerAddressMode::eRepeat,
                                       vk::SamplerMipmapMode::eNearest);
}

SamplerHandle SamplerPool::nearest_clamp_to_edge() {
    return for_filter_and_address_mode(vk::Filter::eNearest, vk::Filter::eNearest,
                                       vk::SamplerAddressMode::eClampToEdge,
                                       vk::SamplerMipmapMode::eNearest);
}

SamplerHandle SamplerPool::nearest_clamp_to_border(const vk::BorderColor border_color) {
    return for_filter_and_address_mode(vk::Filter::eNearest, vk::Filter::eNearest,
                                       vk::SamplerAddressMode::eClampToBorder,
                                       vk::SamplerMipmapMode::eNearest, true, border_color);
}

} // namespace merian
