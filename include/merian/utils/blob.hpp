#pragma once

#include <cassert>
#include <memory>
#include <string>
#include <vector>

namespace merian {

class Blob;
using BlobHandle = std::shared_ptr<Blob>;

class Blob {
  public:
    virtual void* get_data() = 0;

    template <typename T> T* get_data() {
        return reinterpret_cast<T*>(get_data());
    }

    // in bytes
    virtual std::size_t get_size() = 0;
};

class StringBlob : public Blob {
  public:
    StringBlob(const std::string& data) : data(data) {}

    StringBlob(std::string&& data) : data(data) {}

    StringBlob(const std::size_t size_bytes) : data() {
        data.resize(size_bytes);
    }

    void* get_data() override {
        return data.data();
    }

    std::size_t get_size() override {
        return data.size();
    }

    std::string& get_string() {
        return data;
    }

  private:
    std::string data;
};

template <typename T> class VectorBlob : public Blob {
  public:
    VectorBlob(const std::size_t size_bytes) : data(size_bytes / sizeof(T)) {
        assert(size_bytes % sizeof(T) == 0);
    }

    VectorBlob(const std::vector<T>& data) : data(data) {}

    VectorBlob(std::vector<T>&& data) : data(data) {}

    void* get_data() override {
        return data.data();
    }

    std::size_t get_size() override {
        return data.size();
    }

    std::vector<T>& get_vector() {
        return data;
    }

  private:
    std::vector<T> data;
};

} // namespace merian
