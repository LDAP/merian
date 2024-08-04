#pragma once

#include "merian/vk/context.hpp"

#include <memory>

namespace merian {
class Semaphore : public std::enable_shared_from_this<Semaphore> {
  public:
    Semaphore(const ContextHandle& context, const vk::SemaphoreTypeCreateInfo& type_create_info)
        : context(context) {
        vk::SemaphoreCreateInfo create_info{{}, &type_create_info};
        semaphore = context->device.createSemaphore(create_info);
    }

    virtual ~Semaphore() {
        context->device.destroySemaphore(semaphore);
    }

    Semaphore(const Semaphore&) = delete;

    operator const vk::Semaphore&() const {
        return semaphore;
    }

    const vk::Semaphore& operator*() {
      return semaphore;
    }

  protected:
    const ContextHandle context;
    vk::Semaphore semaphore;
};

using SemaphoreHandle = std::shared_ptr<Semaphore>;

} // namespace merian
