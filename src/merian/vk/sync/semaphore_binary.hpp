#pragma once

#include "merian/vk/sync/semaphore.hpp"

namespace merian {
class BinarySemaphore : public Semaphore {
  public:
    BinarySemaphore(const SharedContext& context)
        : Semaphore(context, {vk::SemaphoreType::eBinary, 0}) {}
};

using BinarySemaphoreHandle = std::shared_ptr<BinarySemaphore>;
} // namespace merian