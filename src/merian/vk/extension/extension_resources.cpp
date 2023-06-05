#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/memory_allocator_vma.hpp"

namespace merian {

void ExtensionResources::on_context_created(const SharedContext context) {
    weak_context = context;
}

void ExtensionResources::on_destroy_context() {
}

} // namespace merian
