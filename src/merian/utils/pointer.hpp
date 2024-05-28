#pragma once

#include <memory>

namespace merian {

// returns true if test_ptr can be casted to a shared_ptr of one of TestTypes.
template <typename PtrType, typename... TestTypes>
static bool test_ptr_types(const std::shared_ptr<PtrType>& test_ptr) {
    return (... | (std::dynamic_pointer_cast<TestTypes>(test_ptr) != nullptr));
}

// tests the cast in debug mode
template <typename OutType, typename InType>
std::shared_ptr<OutType> debugable_ptr_cast(const std::shared_ptr<InType>&& ptr) {
    assert(std::dynamic_pointer_cast<OutType>(ptr));
    return std::static_pointer_cast<OutType>(ptr);
}

// tests the cast in debug mode
template <typename OutType, typename InType>
std::shared_ptr<OutType> debugable_ptr_cast(const std::shared_ptr<InType>& ptr) {
    assert(std::dynamic_pointer_cast<OutType>(ptr));
    return std::static_pointer_cast<OutType>(ptr);
}

} // namespace merian
