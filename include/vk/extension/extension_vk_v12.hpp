#pragma once

#include "vk/extension/extension.hpp"

class ExtensionVkV12 : public Extension {
  public:
    ExtensionVkV12() {
        v12f.uniformAndStorageBuffer8BitAccess = VK_TRUE;
        v12f.descriptorIndexing = VK_TRUE;
        v12f.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        v12f.runtimeDescriptorArray = VK_TRUE;
        v12f.bufferDeviceAddress = VK_TRUE;
    }
    ~ExtensionVkV12() {}
    std::string name() const override {
        return "ExtensionVkV12";
    }
    void* on_create_device(void* p_next) override {
        v12f.pNext = p_next;
        return &v12f;
    }

  private:
    vk::PhysicalDeviceVulkan12Features v12f;
};
