#pragma once

#include "merian/utils/input_listener.hpp"
#include "merian/utils/vector_matrix.hpp"

#include <memory>
#include <optional>

namespace merian {

// Over the panels, clicks and scrolling belong to the panels whatever the modifiers.
class PathDebugInput : public InputListener {
  public:
    static constexpr int PRIORITY = 5;
    // the node's own UI: below a graph UI drawn over it, above the picking
    static constexpr int UI_PRIORITY = 8;
    // above any UI, which consumes the key events while it has the keyboard
    static constexpr int OBSERVER_PRIORITY = 1000;
    static constexpr int PICK_MODIFIER = InputController::ALT;
    static constexpr int WIDEN_MODIFIER = InputController::CONTROL;

    struct Click {
        float2 pos;
        // with the pick modifier held
        bool pick;
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

    // Empty while a listener above has the cursor.
    std::optional<float2> cursor() const;
    std::optional<Click> take_click();
    double take_panel_scroll();
    double take_widen_scroll();
    // Escape pressed, whoever consumed it.
    bool take_escape();
    void set_consume_escape(bool consume);

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
        bool escape_pressed = false;
    };

    bool over_panels() const;

    std::shared_ptr<Observer> observed = std::make_shared<Observer>();

    std::optional<Click> click;
    float2 panel_lower{};
    float2 panel_upper{};
    double panel_scroll = 0.;
    double widen_scroll = 0.;
    bool swallow_release = false;
    bool consume_escape = false;
};

} // namespace merian
