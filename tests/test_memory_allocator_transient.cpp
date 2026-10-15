#include <gtest/gtest.h>

#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_validation_layers.hpp"
#include "merian/vk/memory/memory_allocator_transient.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <bit>
#include <random>

using namespace merian;

namespace {

constexpr vk::DeviceSize MIB = vk::DeviceSize{1024} * 1024;

class TransientMemoryAllocatorTest : public ::testing::Test {
  protected:
    static ContextHandle context;
    static ResourceAllocatorHandle allocator;

    static void SetUpTestSuite() {
        context = Context::create({
            .context_extensions = {ExtensionVkValidationLayers::name, ExtensionResources::name},
            .application_name = "test-memory-allocator-transient",
        });
        allocator = context->get_context_extension<ExtensionResources>()->resource_allocator();
    }

    static void TearDownTestSuite() {
        context->get_device()->get_device().waitIdle();
        allocator.reset();
        context.reset();
    }

    // In use from first to last of 16 positions.
    static TransientMemoryAllocator::Lifetime span(const uint32_t first, const uint32_t last) {
        TransientMemoryAllocator::Lifetime lifetime(16);
        lifetime.add(first, last - first + 1);
        return lifetime;
    }

    static TransientMemoryAllocatorHandle create_allocator() {
        return TransientMemoryAllocator::create(allocator->get_memory_allocator());
    }

    static BufferHandle create_buffer(const TransientMemoryScopeHandle& scope,
                                      const vk::DeviceSize size) {
        return scope->create_buffer(
            vk::BufferCreateInfo{{}, size, vk::BufferUsageFlagBits::eStorageBuffer});
    }

    static vk::ImageCreateInfo image_create_info(const uint32_t width, const uint32_t height) {
        return {{},
                vk::ImageType::e2D,
                vk::Format::eR16G16B16A16Sfloat,
                {width, height, 1},
                1,
                1,
                vk::SampleCountFlagBits::e1,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled};
    }

    static ImageHandle create_image(const TransientMemoryScopeHandle& scope,
                                    const uint32_t width,
                                    const uint32_t height) {
        return scope->create_image(image_create_info(width, height));
    }
};

ContextHandle TransientMemoryAllocatorTest::context;
ResourceAllocatorHandle TransientMemoryAllocatorTest::allocator;

// Selects another memory type for buffers than its backing allocator.
class BufferMemoryTypeAllocator : public MemoryAllocator {
  public:
    BufferMemoryTypeAllocator(const MemoryAllocatorHandle& backing, const uint32_t memory_type)
        : MemoryAllocator(backing->get_context()), backing(backing), memory_type(memory_type) {}

    MemoryAllocationHandle allocate_memory(const vk::MemoryPropertyFlags required_flags,
                                           const vk::MemoryRequirements& requirements,
                                           const std::string& debug_name,
                                           const MemoryMappingType mapping_type,
                                           const vk::MemoryPropertyFlags preferred_flags,
                                           const bool dedicated,
                                           const float dedicated_priority) override {
        return backing->allocate_memory(required_flags, requirements, debug_name, mapping_type,
                                        preferred_flags, dedicated, dedicated_priority);
    }

    std::optional<uint32_t> find_memory_type(const vk::BufferCreateInfo& /*buffer_create_info*/,
                                             const MemoryMappingType /*mapping_type*/) override {
        return memory_type;
    }

    std::optional<uint32_t> find_memory_type(const vk::ImageCreateInfo& image_create_info,
                                             const MemoryMappingType mapping_type) override {
        return backing->find_memory_type(image_create_info, mapping_type);
    }

  private:
    const MemoryAllocatorHandle backing;
    const uint32_t memory_type;
};

} // namespace

// Lower bound: the largest two neighbors alive at once, 32 + 32 MIB.
TEST_F(TransientMemoryAllocatorTest, ChainReachesLowerBound) {
    const TransientMemoryAllocatorHandle transient = create_allocator();
    const std::vector<vk::DeviceSize> sizes = {8 * MIB, 16 * MIB, 32 * MIB, 32 * MIB};
    std::vector<TransientMemoryScopeHandle> scopes;
    std::vector<BufferHandle> buffers;
    for (uint32_t i = 0; i < sizes.size(); i++) {
        scopes.push_back(transient->create_scope(span(i, i + 1)));
        buffers.push_back(create_buffer(scopes.back(), sizes[i]));
    }
    transient->commit();

    ASSERT_EQ(transient->get_blocks().size(), 1);
    EXPECT_EQ(transient->get_blocks()[0].size, 64 * MIB);
    EXPECT_EQ(transient->get_blocks()[0].requested_size, 88 * MIB);
    for (uint32_t i = 0; i < sizes.size(); i++) {
        ASSERT_EQ(scopes[i]->get_placements().size(), 1);
        EXPECT_EQ(scopes[i]->get_placements()[0].size, sizes[i]);
        EXPECT_TRUE(buffers[i]->get_memory());
    }
}

TEST_F(TransientMemoryAllocatorTest, DisjointLifetimesShareMemory) {
    const TransientMemoryAllocatorHandle transient = create_allocator();
    const TransientMemoryScopeHandle a = transient->create_scope(span(0, 0));
    const TransientMemoryScopeHandle b = transient->create_scope(span(1, 1));
    const TransientMemoryScopeHandle c = transient->create_scope(span(1, 2));
    const std::vector<BufferHandle> buffers = {create_buffer(a, 4 * MIB), create_buffer(b, 4 * MIB),
                                               create_buffer(c, 4 * MIB)};
    transient->commit();

    EXPECT_EQ(a->get_placements()[0].offset, b->get_placements()[0].offset);
    EXPECT_NE(b->get_placements()[0].offset, c->get_placements()[0].offset);
    EXPECT_EQ(transient->get_blocks()[0].size, 8 * MIB);
    EXPECT_TRUE(a->get_placements()[0].aliased);
    EXPECT_TRUE(b->get_placements()[0].aliased);
    EXPECT_FALSE(c->get_placements()[0].aliased);
}

// Memory is reused if the block fills at least half of it.
TEST_F(TransientMemoryAllocatorTest, ReusesMemoryThatFits) {
    const auto commit_buffer = [](const vk::DeviceSize size,
                                  const std::vector<MemoryAllocationHandle>& reusable) {
        const TransientMemoryAllocatorHandle transient = create_allocator();
        const BufferHandle buffer = create_buffer(transient->create_scope(span(0, 0)), size);
        transient->commit(reusable);
        return buffer->get_memory();
    };
    const MemoryAllocationHandle memory = commit_buffer(8 * MIB, {});
    ASSERT_TRUE(memory);

    EXPECT_EQ(commit_buffer(8 * MIB, {memory}), memory);
    EXPECT_EQ(commit_buffer(4 * MIB, {memory}), memory);
    EXPECT_NE(commit_buffer(2 * MIB, {memory}), memory);
    EXPECT_NE(commit_buffer(16 * MIB, {memory}), memory);
}

TEST_F(TransientMemoryAllocatorTest, RandomPlacementsAreValid) {
    std::mt19937 rng(42);
    const vk::DeviceSize page = context->get_physical_device()
                                    ->get_properties()
                                    .get_properties()
                                    .limits.bufferImageGranularity;

    for (uint32_t test = 0; test < 100; test++) {
        const TransientMemoryAllocatorHandle transient = create_allocator();
        const uint32_t count = std::uniform_int_distribution<uint32_t>(1, 40)(rng);
        const uint32_t position_count = std::uniform_int_distribution<uint32_t>(1, 100)(rng);

        struct Resource {
            TransientMemoryScopeHandle scope;
            std::vector<bool> alive;
            vk::MemoryRequirements requirements;
            ImageHandle image;
            BufferHandle buffer;
        };
        std::vector<Resource> resources;
        for (uint32_t i = 0; i < count; i++) {
            Resource& resource = resources.emplace_back();
            resource.alive.assign(position_count, false);
            TransientMemoryAllocator::Lifetime lifetime(position_count);
            // one or two intervals, possibly wrapping
            const uint32_t interval_count = std::uniform_int_distribution<uint32_t>(1, 2)(rng);
            for (uint32_t interval = 0; interval < interval_count; interval++) {
                const uint32_t first =
                    std::uniform_int_distribution<uint32_t>(0, position_count - 1)(rng);
                const uint32_t length =
                    std::uniform_int_distribution<uint32_t>(1, position_count)(rng);
                lifetime.add(first, length);
                for (uint32_t p = 0; p < length; p++) {
                    resource.alive[(first + p) % position_count] = true;
                }
            }
            for (uint32_t p = 0; p < position_count; p++) {
                EXPECT_EQ(lifetime.contains(p), resource.alive[p]);
            }
            resource.scope = transient->create_scope(lifetime);
            if (std::uniform_int_distribution<uint32_t>(0, 3)(rng) == 0) {
                resource.image = create_image(resource.scope,
                                              std::uniform_int_distribution<uint32_t>(1, 512)(rng),
                                              std::uniform_int_distribution<uint32_t>(1, 512)(rng));
                resource.requirements = resource.image->get_memory_requirements();
            } else {
                resource.buffer = create_buffer(
                    resource.scope, std::uniform_int_distribution<vk::DeviceSize>(1, 2 * MIB)(rng));
                resource.requirements = resource.buffer->get_memory_requirements();
            }
        }
        transient->commit();

        const auto& blocks = transient->get_blocks();
        std::vector<vk::DeviceSize> block_end(blocks.size(), 0);
        std::vector<vk::DeviceSize> block_requested(blocks.size(), 0);
        for (uint32_t i = 0; i < count; i++) {
            const Resource& resource = resources[i];
            ASSERT_EQ(resource.scope->get_placements().size(), 1);
            const auto p = resource.scope->get_placements()[0];
            ASSERT_LT(p.block_index, blocks.size());
            EXPECT_EQ(p.size, resource.requirements.size);
            EXPECT_EQ(p.offset % resource.requirements.alignment, 0);
            EXPECT_EQ(p.offset % page, 0);
            EXPECT_TRUE(resource.requirements.memoryTypeBits &
                        (1u << blocks[p.block_index].memory_type_index));
            block_end[p.block_index] = std::max(block_end[p.block_index], p.offset + p.size);
            block_requested[p.block_index] += p.size;

            for (uint32_t j = 0; j < i; j++) {
                const auto q = resources[j].scope->get_placements()[0];
                bool alive_together = false;
                for (uint32_t position = 0; position < position_count; position++) {
                    alive_together |= resource.alive[position] && resources[j].alive[position];
                }
                EXPECT_EQ(
                    resource.scope->get_lifetime().overlaps(resources[j].scope->get_lifetime()),
                    alive_together);
                const bool overlap = p.block_index == q.block_index &&
                                     p.offset < q.offset + q.size && q.offset < p.offset + p.size;
                EXPECT_FALSE(alive_together && overlap)
                    << "test " << test << ": resources " << i << " and " << j;
            }

            if (resource.image) {
                EXPECT_TRUE(resource.image->get_memory());
                EXPECT_TRUE(allocator->create_image_view(resource.image,
                                                         resource.image->make_view_create_info()));
            } else {
                EXPECT_TRUE(resource.buffer->get_memory());
            }
        }
        for (uint32_t b = 0; b < blocks.size(); b++) {
            EXPECT_GE(blocks[b].size, block_end[b]);
            EXPECT_LT(blocks[b].size, block_end[b] + page);
            EXPECT_EQ(blocks[b].requested_size, block_requested[b]);
        }
    }
}

TEST_F(TransientMemoryAllocatorTest, CreateAfterCommitThrows) {
    const TransientMemoryAllocatorHandle transient = create_allocator();
    const TransientMemoryScopeHandle scope = transient->create_scope(span(0, 0));
    const BufferHandle buffer = create_buffer(scope, MIB);
    transient->commit();
    EXPECT_THROW(create_buffer(scope, MIB), std::logic_error);
}

TEST_F(TransientMemoryAllocatorTest, MappedMemoryIsNotPlaced) {
    const TransientMemoryAllocatorHandle transient = create_allocator();
    const TransientMemoryScopeHandle scope = transient->create_scope(span(0, 0));
    const BufferHandle mapped =
        scope->create_buffer(vk::BufferCreateInfo{{}, MIB, vk::BufferUsageFlagBits::eTransferSrc},
                             MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE);
    EXPECT_TRUE(mapped->get_memory());
    transient->commit();
    EXPECT_TRUE(scope->get_placements().empty());
    EXPECT_TRUE(transient->get_blocks().empty());
    ASSERT_EQ(scope->get_fallbacks().size(), 1);
    EXPECT_EQ(scope->get_fallbacks()[0].reason,
              TransientMemoryAllocator::Fallback::Reason::HOST_ACCESS);
}

TEST_F(TransientMemoryAllocatorTest, FullLifetimeNeverShares) {
    const TransientMemoryAllocatorHandle transient = create_allocator();
    const TransientMemoryScopeHandle full =
        transient->create_scope(TransientMemoryAllocator::Lifetime(16, true));
    const TransientMemoryScopeHandle a = transient->create_scope(span(0, 0));
    const TransientMemoryScopeHandle b = transient->create_scope(span(15, 15));
    const std::vector<BufferHandle> buffers = {
        create_buffer(full, 4 * MIB), create_buffer(a, 4 * MIB), create_buffer(b, 4 * MIB)};
    transient->commit();

    EXPECT_FALSE(full->get_placements()[0].aliased);
    EXPECT_EQ(a->get_placements()[0].offset, b->get_placements()[0].offset);
    EXPECT_NE(a->get_placements()[0].offset, full->get_placements()[0].offset);
    EXPECT_EQ(transient->get_blocks()[0].size, 8 * MIB);
}

// Two iterations of four layers: A is written in layer 1 and read there one iteration later, T is
// written in layer 2 and read in layer 3. Each copy of T fits into the copy of A not in use.
TEST_F(TransientMemoryAllocatorTest, CyclicLifetimesShareMemory) {
    const TransientMemoryAllocatorHandle transient = create_allocator();
    std::vector<TransientMemoryAllocator::Lifetime> lifetimes(
        4, TransientMemoryAllocator::Lifetime(8));
    lifetimes[0].add(1, 5);
    lifetimes[1].add(5, 5);
    lifetimes[2].add(2, 2);
    lifetimes[3].add(6, 2);
    std::vector<TransientMemoryScopeHandle> scopes;
    std::vector<BufferHandle> buffers;
    for (const TransientMemoryAllocator::Lifetime& lifetime : lifetimes) {
        scopes.push_back(transient->create_scope(lifetime));
        buffers.push_back(create_buffer(scopes.back(), 4 * MIB));
    }
    transient->commit();

    EXPECT_EQ(transient->get_blocks()[0].size, 8 * MIB);
    EXPECT_NE(scopes[0]->get_placements()[0].offset, scopes[1]->get_placements()[0].offset);
    EXPECT_EQ(scopes[2]->get_placements()[0].offset, scopes[1]->get_placements()[0].offset);
    EXPECT_EQ(scopes[3]->get_placements()[0].offset, scopes[0]->get_placements()[0].offset);
}

// Three buffers in use together need more than the largest allocation the device allows.
TEST_F(TransientMemoryAllocatorTest, PlacementsSpillIntoFurtherBlocks) {
    const vk::DeviceSize max_allocation_size = context->get_physical_device()
                                                   ->get_properties()
                                                   .get_maintenance3_properties()
                                                   .maxMemoryAllocationSize;
    const vk::DeviceSize limit =
        std::min(TransientMemoryAllocator::MAX_BLOCK_SIZE, max_allocation_size);
    const vk::DeviceSize size = limit / 8 * 3;
    const std::optional<uint32_t> memory_type = allocator->get_memory_allocator()->find_memory_type(
        vk::BufferCreateInfo{{}, size, vk::BufferUsageFlagBits::eStorageBuffer});
    ASSERT_TRUE(memory_type);
    const vk::PhysicalDeviceMemoryProperties& memory_properties =
        context->get_physical_device()->get_memory_properties().memoryProperties;
    if (memory_properties.memoryHeaps[memory_properties.memoryTypes[*memory_type].heapIndex].size <
        3 * size) {
        GTEST_SKIP() << "the heap cannot hold three buffers";
    }
    std::vector<TransientMemoryScopeHandle> scopes;
    std::vector<BufferHandle> buffers;
    const auto commit = [&](std::vector<MemoryAllocationHandle> reusable) {
        const TransientMemoryAllocatorHandle transient = create_allocator();
        scopes.clear();
        buffers.clear();
        for (uint32_t i = 0; i < 3; i++) {
            scopes.push_back(transient->create_scope(span(0, 0)));
            buffers.push_back(create_buffer(scopes.back(), size));
        }
        transient->commit(std::move(reusable));
        return transient;
    };

    const TransientMemoryAllocatorHandle first = commit({});
    ASSERT_EQ(first->get_blocks().size(), 2);
    std::vector<MemoryAllocationHandle> memory;
    for (const TransientMemoryAllocator::Block& block : first->get_blocks()) {
        EXPECT_LE(block.size, limit);
        memory.push_back(block.memory);
    }
    for (uint32_t i = 0; i < 3; i++) {
        EXPECT_TRUE(buffers[i]->get_memory());
        for (uint32_t j = 0; j < i; j++) {
            const auto p = scopes[i]->get_placements().at(0);
            const auto q = scopes[j]->get_placements().at(0);
            EXPECT_FALSE(p.block_index == q.block_index && p.offset < q.offset + q.size &&
                         q.offset < p.offset + p.size);
        }
    }

    // the next commit fits into the same blocks
    const TransientMemoryAllocatorHandle second = commit(memory);
    ASSERT_EQ(second->get_blocks().size(), 2);
    for (uint32_t i = 0; i < 2; i++) {
        EXPECT_EQ(second->get_blocks()[i].memory, memory[i]);
    }
}

TEST_F(TransientMemoryAllocatorTest, BlocksOfSeveralMemoryTypes) {
    const MemoryAllocatorHandle& backing = allocator->get_memory_allocator();
    const uint32_t image_memory_type = *backing->find_memory_type(image_create_info(64, 64));
    const uint32_t other_memory_types = backing
                                            ->get_buffer_memory_requirements(vk::BufferCreateInfo{
                                                {}, MIB, vk::BufferUsageFlagBits::eStorageBuffer})
                                            .memoryTypeBits &
                                        ~(1u << image_memory_type);
    if (other_memory_types == 0) {
        GTEST_SKIP() << "buffers support only the memory type of images";
    }
    const uint32_t buffer_memory_type = std::countr_zero(other_memory_types);

    const TransientMemoryAllocatorHandle transient = TransientMemoryAllocator::create(
        std::make_shared<BufferMemoryTypeAllocator>(backing, buffer_memory_type));
    const std::vector<TransientMemoryScopeHandle> scopes = {transient->create_scope(span(0, 0)),
                                                            transient->create_scope(span(0, 0)),
                                                            transient->create_scope(span(0, 0))};
    const ImageHandle image = create_image(scopes[0], 64, 64);
    const BufferHandle first = create_buffer(scopes[1], MIB);
    const BufferHandle second = create_buffer(scopes[2], MIB);
    transient->commit();

    const auto& blocks = transient->get_blocks();
    ASSERT_EQ(blocks.size(), 2);
    const uint32_t image_block = scopes[0]->get_placements().at(0).block_index;
    const uint32_t buffer_block = scopes[1]->get_placements().at(0).block_index;
    EXPECT_NE(image_block, buffer_block);
    EXPECT_EQ(scopes[2]->get_placements().at(0).block_index, buffer_block);
    EXPECT_EQ(blocks[image_block].memory_type_index, image_memory_type);
    EXPECT_EQ(blocks[buffer_block].memory_type_index, buffer_memory_type);
    EXPECT_EQ(blocks[buffer_block].requested_size, 2 * MIB);
}

TEST_F(TransientMemoryAllocatorTest, ResourceOutlivesScope) {
    const TransientMemoryAllocatorHandle transient = create_allocator();
    const BufferHandle buffer = create_buffer(transient->create_scope(span(0, 0)), MIB);
    transient->commit();
    EXPECT_TRUE(buffer->get_memory());
    EXPECT_EQ(transient->get_blocks().size(), 1);
}
