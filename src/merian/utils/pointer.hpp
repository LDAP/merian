#pragma once

#include <memory>

namespace merian {

// returns true if test_ptr can be casted to a shared_ptr of one of TestTypes.
template <typename PtrType, typename ...TestTypes>
static bool test_ptr_types(const std::shared_ptr<PtrType>& test_ptr) {
    return (... | (std::dynamic_pointer_cast<TestTypes>(test_ptr) != nullptr));
}

// does a 
#define debuggable_ptr_cast


} // namespace merian
