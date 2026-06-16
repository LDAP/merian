#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace merian {

// A value lazily rebuilt from its inputs: get() rebuilds it only when an input's version changed.
// The returned shared_ptr stays valid even after a later rebuild.
template <typename T> class Versioned {
  public:
    using Ref = std::shared_ptr<T>;

    Versioned() = default;

    explicit Versioned(std::function<Ref()> build)
        : state(std::make_shared<State>(std::move(build))) {}

    template <typename U> Versioned& depends_on(const Versioned<U>& dep) {
        return depends_on([dep] {
            dep.get();
            return dep.version();
        });
    }

    Versioned& depends_on(std::function<uint64_t()> input_version) {
        state->inputs.push_back(std::move(input_version));
        return *this;
    }

    const Ref& get() const {
        const uint64_t inputs = state->inputs_version();
        if (!state->current || inputs != state->built_at) {
            state->current = state->build();
            state->built_at = inputs;
            state->version++;
        }
        return state->current;
    }

    // rebuilds like get(); use peek() to access without rebuilding
    const Ref& operator->() const {
        return get();
    }

    // null before the first get()
    const Ref& peek() const {
        return state->current;
    }

    uint64_t version() const {
        return state->version;
    }

  private:
    struct State {
        explicit State(std::function<Ref()> build) : build(std::move(build)) {}

        uint64_t inputs_version() const {
            uint64_t sum = 0;
            for (const auto& input : inputs) {
                sum += input();
            }
            return sum;
        }

        std::function<Ref()> build;
        std::vector<std::function<uint64_t()>> inputs;
        Ref current;
        uint64_t built_at = 0;
        uint64_t version = 0;
    };

    // shared so a copy captured in a downstream builder sees the same cache
    std::shared_ptr<State> state;
};

} // namespace merian
