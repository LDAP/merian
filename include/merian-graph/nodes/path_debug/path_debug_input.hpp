#pragma once

#include "merian/utils/input_listener.hpp"
#include "merian/utils/vector_matrix.hpp"

#include <memory>
#include <optional>

namespace merian {

// Every binding needs the modifier held, so the scene keeps its own; over the panels, clicks and
// scrolling belong to the panels whatever the modifiers.
class PathDebugInput : public InputListener {
  public:
    static constexpr int PRIORITY = 5;
    // the node's own UI: below a graph UI drawn over it, above the picking
    static constexpr int UI_PRIORITY = 8;
    // above any UI, which consumes the key events while it has the keyboard
    static constexpr int OBSERVER_PRIORITY = 1000;

    struct Click {
        float2 pos;
        // with the modifier held
        bool select;
    };

    // What arrived since the last consume_events().
    struct Events {
        std::optional<Click> click;
        double panel_scroll = 0.;
        double radius_scroll = 0.;
        // Escape with the modifier held, whoever consumed it
        bool clear = false;
    };

    bool on_cursor(InputController& controller, double xpos, double ypos) override;
    bool on_key(InputController& controller,
                InputController::Key key,
                InputController::KeyStatus action,
                int mods) override;
    bool on_mouse_button(InputController& controller,
                         InputController::MouseButton button,
                         InputController::KeyStatus status) override;
    bool on_scroll(InputController& controller, double xoffset, double yoffset) override;

    // Tracks the held modifiers and the cursor; never consumes an event.
    InputListenerHandle observer() const;

    // In window coordinates; empty without panels.
    void set_panel_area(const float2& lower, const float2& upper);

    // InputController::ModKey bits.
    void set_modifier(int modifier);

    // Empty while a listener above has the cursor.
    std::optional<float2> cursor() const;
    Events consume_events();

  private:
    class Observer : public InputListener {
      public:
        bool on_cursor(InputController& controller, double xpos, double ypos) override;
        bool on_key(InputController& controller,
                    InputController::Key key,
                    InputController::KeyStatus action,
                    int mods) override;

        int held_mods = 0;
        bool cursor_seen = false;
        float2 cursor_pos{};
        // the last cursor event did not reach PRIORITY
        bool cursor_taken = false;
        // the modifiers held at the last Escape press
        std::optional<int> escape_mods;
    };

    bool over_panels() const;
    bool modifier_held(int mods) const;

    std::shared_ptr<Observer> observed = std::make_shared<Observer>();

    Events events;
    float2 panel_lower{};
    float2 panel_upper{};

    int modifier = InputController::ALT;
    bool swallow_release = false;
};

} // namespace merian
