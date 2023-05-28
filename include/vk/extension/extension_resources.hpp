#include "vk/extension/extension.hpp"

#include "vk/memory/resource_allocator.hpp"
#include "vk_mem_alloc.h"

namespace merian {

    class ExtensionResources : public Extension {
    public:
        ExtensionResources() : Extension("ExtensionResources") {}
        ~ExtensionResources() {}

        void on_context_created(const Context& context) override;
        void on_destroy_context(const Context& context) override;

        MemoryAllocator& memory_allocator() {
            return *_memory_allocator;
        }
        ResourceAllocator& resource_allocator() {
            return *_resource_allocator;
        }
        SamplerPool& sampler_pool() {
            return *_sampler_pool;
        }

    private:
        VmaAllocator vma_allocator = VK_NULL_HANDLE;
        MemoryAllocator* _memory_allocator;
        ResourceAllocator* _resource_allocator;
        SamplerPool* _sampler_pool;
    };

}
