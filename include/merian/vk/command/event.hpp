#pragma once

#include "merian/vk/context.hpp"

namespace merian {

class Event : public std::enable_shared_from_this<Event>, public Object {

  public:
    Event() = delete;

    Event(const ContextHandle& context, const vk::EventCreateFlags flags = {}) : context(context) {
        vk::EventCreateInfo info{flags};
        event = context->device.createEvent(info);
    }

    ~Event() {
        context->device.destroyEvent(event);
    }

    operator const vk::Event&() const {
        return event;
    }

    const vk::Event& get_event() const {
        return event;
    }

  private:
    const ContextHandle context;
    vk::Event event;
};

using EventHandle = std::shared_ptr<Event>;

} // namespace merian
