#pragma once

#include <functional>

template<typename T, typename ...Args>
void run_if_set(std::function<T(Args...)> function, Args... args) {
    if (function)
        function(args...);
}
