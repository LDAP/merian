#pragma once

#include <vector>

namespace merian {

/**
 * @brief      A ring buffer that stores every element twice to allow for consecutive pointers
 * without iterators.
 *
 * @tparam     T     the type, must be compatibe with std::vector
 */
template <typename T> class RingBuffer {

  public:
    RingBuffer(const std::size_t size, const T& value) : ring_size(size), buffer(size * 2, value) {}

    RingBuffer(const std::size_t size) : ring_size(size), buffer(size * 2) {}

    const std::size_t& size() const {
        return ring_size;
    }

    const T& operator[](const std::size_t index) const {
        return buffer[index % ring_size];
    }

    void set(const std::size_t index, const T& value) {
        const std::size_t ring_index = index % ring_size;
        buffer[ring_index] = buffer[ring_index + ring_size] = value;
    }

  private:
    const std::size_t ring_size;
    std::vector<T> buffer;
};

} // namespace merian
