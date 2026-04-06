#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace merian {

/**
 * @brief Base class for objects whose layout/structure can change over time.
 *
 * Provides a version counter and listener mechanism for push-based change propagation.
 * Only structural/layout changes should trigger increment_version() — not per-frame
 * content updates like descriptor writes.
 *
 * Listeners are registered with a shared_ptr guard. When the guard expires, the listener
 * is automatically pruned.
 */
class Versionable {
  public:
    using Version = uint64_t;

    virtual ~Versionable() = default;

    Version get_version() const {
        return version;
    }

    /**
     * @brief Register a listener that is called when this object's version changes.
     *
     * @param guard Shared pointer controlling listener lifetime. When it expires,
     *              the listener is automatically removed during the next notification.
     * @param callback Called when increment_version() is invoked.
     */
    void on_changed(const std::shared_ptr<void>& guard, std::function<void()> callback) {
        listeners.push_back({guard, std::move(callback)});
    }

  protected:
    /**
     * @brief Increment version and notify all live listeners.
     *
     * Prunes listeners whose guard has expired.
     */
    void increment_version() {
        version++;

        std::size_t write = 0;
        for (std::size_t read = 0; read < listeners.size(); read++) {
            if (auto locked = listeners[read].guard.lock()) {
                listeners[read].callback();
                if (write != read) {
                    listeners[write] = std::move(listeners[read]);
                }
                write++;
            }
        }
        listeners.resize(write);
    }

  private:
    Version version = 0;

    struct Listener {
        std::weak_ptr<void> guard;
        std::function<void()> callback;
    };
    std::vector<Listener> listeners;
};

} // namespace merian
