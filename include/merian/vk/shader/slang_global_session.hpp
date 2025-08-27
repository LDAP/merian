#pragma once

#include "slang-com-ptr.h"
#include "slang.h"

namespace merian {

Slang::ComPtr<slang::IGlobalSession> get_global_slang_session();

}
