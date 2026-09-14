#include "ngx.hpp"

#include <fmt/format.h>

#include <filesystem>
#include <vector>

namespace merian {

namespace {

// Identifies merian for over-the-air preset updates. NGX rejects anything but a UUID here.
constexpr const char* PROJECT_ID = "216b2819-6968-4076-b7f0-766f8f3b3ca2";

} // namespace

std::string ngx_result_string(const NVSDK_NGX_Result result) {
    switch (result) {
    case NVSDK_NGX_Result_Success:
        return "success";
    case NVSDK_NGX_Result_FAIL_FeatureNotSupported:
        return "feature not supported on this hardware";
    case NVSDK_NGX_Result_FAIL_PlatformError:
        return "platform error";
    case NVSDK_NGX_Result_FAIL_FeatureAlreadyExists:
        return "feature already exists";
    case NVSDK_NGX_Result_FAIL_FeatureNotFound:
        return "feature not found";
    case NVSDK_NGX_Result_FAIL_InvalidParameter:
        return "invalid parameter";
    case NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall:
        return "scratch buffer too small";
    case NVSDK_NGX_Result_FAIL_NotInitialized:
        return "not initialized";
    case NVSDK_NGX_Result_FAIL_UnsupportedInputFormat:
        return "unsupported input format";
    case NVSDK_NGX_Result_FAIL_RWFlagMissing:
        return "resource is missing the storage flag";
    case NVSDK_NGX_Result_FAIL_MissingInput:
        return "a required input is missing";
    case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature:
        return "unable to initialize feature";
    case NVSDK_NGX_Result_FAIL_OutOfDate:
        return "the feature library is out of date";
    case NVSDK_NGX_Result_FAIL_OutOfGPUMemory:
        return "out of device memory";
    case NVSDK_NGX_Result_FAIL_UnsupportedFormat:
        return "unsupported format";
    case NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath:
        return "cannot write to the application data path";
    case NVSDK_NGX_Result_FAIL_UnsupportedParameter:
        return "unsupported parameter";
    case NVSDK_NGX_Result_FAIL_Denied:
        return "denied";
    case NVSDK_NGX_Result_FAIL_NotImplemented:
        return "not implemented";
    default:
        return fmt::format("unknown NGX result 0x{:x}", static_cast<uint32_t>(result));
    }
}

NVSDK_NGX_PerfQuality_Value ngx_perf_quality(const DLSSQuality quality) {
    switch (quality) {
    case DLSSQuality::UltraPerformance:
        return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
    case DLSSQuality::Performance:
        return NVSDK_NGX_PerfQuality_Value_MaxPerf;
    case DLSSQuality::Balanced:
        return NVSDK_NGX_PerfQuality_Value_Balanced;
    case DLSSQuality::Quality:
        return NVSDK_NGX_PerfQuality_Value_MaxQuality;
    case DLSSQuality::DLAA:
        return NVSDK_NGX_PerfQuality_Value_DLAA;
    }
    return NVSDK_NGX_PerfQuality_Value_MaxQuality;
}

NVSDK_NGX_Application_Identifier ngx_application_identifier() {
    NVSDK_NGX_Application_Identifier identifier{};
    identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
    identifier.v.ProjectDesc.ProjectId = PROJECT_ID;
    identifier.v.ProjectDesc.EngineType = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
    identifier.v.ProjectDesc.EngineVersion = MERIAN_VERSION;
    return identifier;
}

const std::wstring& ngx_application_data_path() {
    static const std::wstring path = [] {
        const std::filesystem::path dir = std::filesystem::temp_directory_path() / "merian-ngx";
        std::error_code error;
        std::filesystem::create_directories(dir, error);
        return dir.wstring();
    }();
    return path;
}

const NVSDK_NGX_FeatureCommonInfo& ngx_feature_common_info() {
    static const std::vector<std::wstring> search_paths = [] {
        std::vector<std::wstring> paths;
#ifdef MERIAN_NGX_SNIPPET_DIR
        paths.emplace_back(std::filesystem::path(MERIAN_NGX_SNIPPET_DIR).wstring());
#endif
        return paths;
    }();
    static const std::vector<const wchar_t*> search_path_ptrs = [] {
        std::vector<const wchar_t*> ptrs;
        for (const std::wstring& path : search_paths) {
            ptrs.emplace_back(path.c_str());
        }
        return ptrs;
    }();

    static const NVSDK_NGX_FeatureCommonInfo info = [] {
        NVSDK_NGX_FeatureCommonInfo common{};
        common.PathListInfo.Path = search_path_ptrs.data();
        common.PathListInfo.Length = static_cast<uint32_t>(search_path_ptrs.size());
        common.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_OFF;
        return common;
    }();
    return info;
}

} // namespace merian
