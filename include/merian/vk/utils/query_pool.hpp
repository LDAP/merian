#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_vk_core.hpp"
#include "merian/vk/utils/check_result.hpp"

#include <memory>

namespace merian {

template <vk::QueryType QUERY_TYPE> class QueryPool;
template <vk::QueryType QUERY_TYPE> using QueryPoolHandle = std::shared_ptr<QueryPool<QUERY_TYPE>>;

template <vk::QueryType QUERY_TYPE>
class QueryPool : public std::enable_shared_from_this<QueryPool<QUERY_TYPE>>, public Object {

  public:
    // Creates a query pool and resets it.
    QueryPool(const ContextHandle& context,
              const uint32_t query_count = 1024,
              bool host_reset_after_creation = false)
        : context(context), query_count(query_count) {

        const vk::QueryPoolCreateInfo create_info({}, QUERY_TYPE, query_count);
        query_pool = context->get_device()->get_device().createQueryPool(create_info);

        if (host_reset_after_creation) {
            reset();
        }
    }

    ~QueryPool() {
        context->get_device()->get_device().destroyQueryPool(query_pool);
    }

    // ------------------------------------------------------------------

    operator const vk::QueryPool&() const {
        return query_pool;
    }

    const vk::QueryPool& operator*() const {
        return query_pool;
    }

    const vk::QueryPool& get_query_pool() const {
        return query_pool;
    }

    // ------------------------------------------------------------------

    // uses the Vulkan 1.2 hostQueryReset feature to reset the pool,
    // use CommandBuffer::reset_query_pool to use the command buffer to reset.
    void reset(const uint32_t first_query, const uint32_t query_count) const {
        assert(context->get_extension<ExtensionVkCore>());
        assert(context->get_extension<ExtensionVkCore>()
                   ->get_enabled_features()
                   .get_physical_device_features_v12()
                   .hostQueryReset);

        context->get_device()->get_device().resetQueryPool(query_pool, first_query, query_count);
    }

    // uses the Vulkan 1.2 hostQueryReset feature to reset the pool
    // use CommandBuffer::reset_query_pool to use the command buffer to reset.
    void reset() {
        reset(0, query_count);
    }

    const uint32_t& get_query_count() const {
        return query_count;
    }

    template <typename RETURN_TYPE>
    std::vector<RETURN_TYPE> get_query_pool_results(const uint32_t first_query,
                                                    const uint32_t query_count,
                                                    const vk::QueryResultFlags flags = {}) const {
        std::vector<RETURN_TYPE> data(query_count);
        check_result(context->get_device()->get_device().getQueryPoolResults(query_pool, first_query, data.size(),
                                                         sizeof(RETURN_TYPE) * data.size(),
                                                         data.data(), sizeof(RETURN_TYPE), flags),
                     "could not get query results");
        return data;
    }

    template <typename RETURN_TYPE>
    std::vector<RETURN_TYPE> get_query_pool_results(const vk::QueryResultFlags flags = {}) const {
        std::vector<RETURN_TYPE> data(query_count);
        check_result(context->get_device()->get_device().getQueryPoolResults(query_pool, 0, data.size(),
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

  private:
    const ContextHandle context;
    const uint32_t query_count;
    vk::QueryPool query_pool;

  public:
    static QueryPoolHandle<QUERY_TYPE> create(const ContextHandle& context,
                                              const uint32_t query_count = 1024,
                                              bool host_reset_after_creation = false) {
        return std::make_shared<QueryPool>(context, query_count, host_reset_after_creation);
    }
};

} // namespace merian
