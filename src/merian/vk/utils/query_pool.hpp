#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/utils/check_result.hpp"

#include <memory>

namespace merian {

template <vk::QueryType QUERY_TYPE>
class QueryPool : public std::enable_shared_from_this<QueryPool<QUERY_TYPE>> {

  public:
    // Creates a query pool and resets it.
    QueryPool(const SharedContext& context,
              const uint32_t query_count = 1024,
              bool reset_after_creation = false)
        : context(context), query_count(query_count) {
        vk::QueryPoolCreateInfo createInfo({}, QUERY_TYPE, query_count);
        query_pool = context->device.createQueryPool(createInfo);
        if (reset_after_creation) {
            reset();
        }
    }

    ~QueryPool() {
        context->device.destroyQueryPool(query_pool);
    }

    void reset(const vk::CommandBuffer& cmd) {
        reset(cmd, 0, query_count);
    }

    void reset() {
        reset(0, query_count);
    }

    void reset(const vk::CommandBuffer& cmd,
               const uint32_t first_query,
               const uint32_t query_count) const {
        cmd.resetQueryPool(query_pool, first_query, query_count);
    }

    // uses the Vulkan 1.2 hostQueryReset feature to reset the pool
    void reset(const uint32_t first_query, const uint32_t query_count) const {
        assert(context->physical_device.features.physical_device_features_v12.hostQueryReset);
        context->device.resetQueryPool(query_pool, first_query, query_count);
    }

    const uint32_t& get_query_count() const {
        return query_count;
    }

    template <typename RETURN_TYPE>
    std::vector<RETURN_TYPE> get_query_pool_results(const uint32_t first_query,
                                                    const uint32_t query_count,
                                                    const vk::QueryResultFlags flags) const {
        std::vector<RETURN_TYPE> data(query_count);
        check_result(context->device.getQueryPoolResults(query_pool, first_query, data.size(),
                                                         sizeof(RETURN_TYPE) * data.size(),
                                                         data.data(), sizeof(RETURN_TYPE), flags),
                     "could not get query results");
        return data;
    }

    std::vector<uint32_t> get_query_pool_results(const uint32_t first_query,
                                                 const uint32_t query_count) const {
        return get_query_pool_results<uint32_t>(first_query, query_count, {});
    }

    std::vector<uint64_t> get_query_pool_results_64(const uint32_t first_query,
                                                    const uint32_t query_count) const {
        return get_query_pool_results<uint64_t>(first_query, query_count,
                                                vk::QueryResultFlagBits::e64);
    }

    std::vector<uint32_t> wait_get_query_pool_results(const uint32_t first_query,
                                                      const uint32_t query_count) const {
        return get_query_pool_results<uint32_t>(first_query, query_count,
                                                vk::QueryResultFlagBits::eWait);
    }

    std::vector<uint64_t> wait_get_query_pool_results_64(const uint32_t first_query,
                                                         const uint32_t query_count) const {
        return get_query_pool_results<uint64_t>(first_query, query_count,
                                                vk::QueryResultFlagBits::e64 |
                                                    vk::QueryResultFlagBits::eWait);
    }

    template <vk::QueryType U = QUERY_TYPE,
              typename = std::enable_if_t<U == vk::QueryType::eTimestamp>>
    void write_timestamp(const vk::CommandBuffer& cmd,
                         const uint32_t query,
                         const vk::PipelineStageFlagBits pipeline_stage =
                             vk::PipelineStageFlagBits::eAllCommands) const {
        cmd.writeTimestamp(pipeline_stage, query_pool, query);
    }

    template <vk::QueryType U = QUERY_TYPE,
              typename = std::enable_if_t<U == vk::QueryType::eTimestamp>>
    void write_timestamp2(const vk::CommandBuffer& cmd,
                          const uint32_t query,
                          const vk::PipelineStageFlagBits2 pipeline_stage =
                              vk::PipelineStageFlagBits2::eAllCommands) const {
        cmd.writeTimestamp2(pipeline_stage, query_pool, query);
    }

  private:
    const SharedContext context;
    const uint32_t query_count;
    vk::QueryPool query_pool;
};
template <vk::QueryType QUERY_TYPE> using QueryPoolHandle = std::shared_ptr<QueryPool<QUERY_TYPE>>;

} // namespace merian
