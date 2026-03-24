#pragma once

#include "merian/vk/memory/memory_allocator.hpp"

#include <memory>

namespace merian {

// Forward declaration to avoid circular include with context.hpp
class Context;
using ContextHandle = std::shared_ptr<Context>;

class MemoryAllocatorProvider {
  public:
    virtual ~MemoryAllocatorProvider() = default;
    virtual MemoryAllocatorHandle create_memory_allocator(const ContextHandle& context) const = 0;
};

} // namespace merian
