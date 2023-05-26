/*
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

#include "vk/sampler/sampler_pool.hpp"
#include <spdlog/spdlog.h>

namespace merian {
//////////////////////////////////////////////////////////////////////////

SamplerPool::~SamplerPool() {
    spdlog::debug("destroy sampler pool");

    if (!sampler_map.empty()) {
      spdlog::debug("SamplerPool still contains {} sampler at destruction", sampler_map.size());
    }

    for (auto it : entries) {
        if (it.sampler) {
            device.destroySampler(it.sampler);
        }
    }
}

vk::Sampler SamplerPool::acquireSampler(const vk::SamplerCreateInfo& createInfo) {
    SamplerState state;
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
        SPDLOG_DEBUG("Create new sampler");

        uint32_t index = 0;
        if (freeIndex != (uint32_t) ~0) {
            index = freeIndex;
            freeIndex = entries[index].nextFreeIndex;
        } else {
            index = (uint32_t)entries.size();
            entries.resize(entries.size() + 1);
        }

        vk::Sampler sampler = device.createSampler(createInfo);

        entries[index].refCount = 1;
        entries[index].sampler = sampler;
        entries[index].state = state;

        state_map.insert({state, index});
        sampler_map.insert({sampler, index});

        return sampler;
    } else {
        SPDLOG_TRACE("Reuse existing sampler");

        entries[it->second].refCount++;
        return entries[it->second].sampler;
    }
}

void SamplerPool::releaseSampler(vk::Sampler sampler) {
    auto it = sampler_map.find(sampler);

    if (it == sampler_map.end()) {
        throw std::runtime_error{"Sampler not found!"};
    }

    uint32_t index = it->second;
    Entry& entry = entries[index];

    assert(entry.sampler == sampler);
    assert(entry.refCount);

    entry.refCount--;

    if (entry.refCount == 0) {
        device.destroySampler(sampler);

        entry.sampler = nullptr;
        entry.nextFreeIndex = freeIndex;
        freeIndex = index;

        state_map.erase(entry.state);
        sampler_map.erase(sampler);
    }
}

} // namespace merian
