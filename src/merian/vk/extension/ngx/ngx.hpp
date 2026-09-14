#pragma once

#include "merian/vk/extension/ngx/dlss.hpp"

#include <vulkan/vulkan.hpp>

// The NGX headers do not include their prerequisites, so the blocks stay in dependency order.
#include "nvsdk_ngx_helpers.h"
#include "nvsdk_ngx_helpers_vk.h"

#include "nvsdk_ngx_helpers_dlssd_d3d.h"
#include "nvsdk_ngx_helpers_dlssd_vk.h"

#include <string>

namespace merian {

std::string ngx_result_string(NVSDK_NGX_Result result);

NVSDK_NGX_PerfQuality_Value ngx_perf_quality(DLSSQuality quality);

NVSDK_NGX_Application_Identifier ngx_application_identifier();

const NVSDK_NGX_FeatureCommonInfo& ngx_feature_common_info();

const std::wstring& ngx_application_data_path();

} // namespace merian
