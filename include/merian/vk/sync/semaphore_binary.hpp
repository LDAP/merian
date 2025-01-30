#pragma once

#include "merian/vk/sync/semaphore.hpp"

namespace merian {

class BinarySemaphore;
using BinarySemaphoreHandle = std::shared_ptr<BinarySemaphore>;

class BinarySemaphore : public Semaphore {
  private:
    BinarySemaphore(const ContextHandle& context)
        : Semaphore(context, {vk::SemaphoreType::eBinary, 0}) {}

  public:
    static BinarySemaphoreHandle create(const ContextHandle& context) {
        return std::shared_ptr<BinarySemaphore>(new BinarySemaphore(context));
    }
};

} // namespace merian
