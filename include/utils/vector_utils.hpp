#pragma once

#include <vector>

template <class T> void insert_all(std::vector<T>& to, std::vector<T>const& from) {
    std::copy(std::begin(from), std::end(from), std::back_inserter(to));
}
