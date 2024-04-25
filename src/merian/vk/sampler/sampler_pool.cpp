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

SamplerHandle SamplerPool::linear_mirrored_repeat() {
    const vk::SamplerCreateInfo info{{},
                                     vk::Filter::eLinear,
                                     vk::Filter::eLinear,
                                     vk::SamplerMipmapMode::eLinear,
                                     vk::SamplerAddressMode::eMirroredRepeat,
                                     vk::SamplerAddressMode::eMirroredRepeat,
                                     vk::SamplerAddressMode::eMirroredRepeat,
                                     {},
                                     false,
                                     context->physical_device.physical_device_properties.properties.limits.maxSamplerAnisotropy,
                                     false,
                                     {},
                                     0.0f,
                                     128.0f,
                                     vk::BorderColor::eIntTransparentBlack,
                                     false};
    return acquire_sampler(info);
}

SamplerHandle SamplerPool::nearest_mirrored_repeat() {
    const vk::SamplerCreateInfo info{{},
                                     vk::Filter::eNearest,
                                     vk::Filter::eNearest,
                                     vk::SamplerMipmapMode::eNearest,
                                     vk::SamplerAddressMode::eMirroredRepeat,
                                     vk::SamplerAddressMode::eMirroredRepeat,
                                     vk::SamplerAddressMode::eMirroredRepeat,
                                     {},
                                     false,
                                     context->physical_device.physical_device_properties.properties.limits.maxSamplerAnisotropy,
                                     false,
                                     {},
                                     0.0f,
                                     128.0f,
                                     vk::BorderColor::eIntTransparentBlack,
                                     false};
    return acquire_sampler(info);
}

SamplerHandle SamplerPool::linear_repeat() {
    const vk::SamplerCreateInfo info{{},
                                     vk::Filter::eLinear,
                                     vk::Filter::eLinear,
                                     vk::SamplerMipmapMode::eLinear,
                                     vk::SamplerAddressMode::eRepeat,
                                     vk::SamplerAddressMode::eRepeat,
                                     vk::SamplerAddressMode::eRepeat,
                                     {},
                                     false,
                                     context->physical_device.physical_device_properties.properties.limits.maxSamplerAnisotropy,
                                     false,
                                     {},
                                     0.0f,
                                     128.0f,
                                     vk::BorderColor::eIntTransparentBlack,
                                     false};
    return acquire_sampler(info);
}


SamplerHandle SamplerPool::nearest_repeat() {
    const vk::SamplerCreateInfo info{{},
                                     vk::Filter::eNearest,
                                     vk::Filter::eNearest,
                                     vk::SamplerMipmapMode::eNearest,
                                     vk::SamplerAddressMode::eRepeat,
                                     vk::SamplerAddressMode::eRepeat,
                                     vk::SamplerAddressMode::eRepeat,
                                     {},
                                     false,
                                     context->physical_device.physical_device_properties.properties.limits.maxSamplerAnisotropy,
                                     false,
                                     {},
                                     0.0f,
                                     128.0f,
                                     vk::BorderColor::eIntTransparentBlack,
                                     false};
    return acquire_sampler(info);
}

} // namespace merian
