#pragma once

#include "glm/glm.hpp"
#include "merian/utils/camera/camera_animator.hpp"
#include "merian/vk/window/glfw_window.hpp"
#include <memory>

namespace merian {

/**
 * @brief      Control a camera using keyboard and mouse.
 *
 * Mouse: Rotate camera
 * Space: Move up
 * Shift: Move down
 *
 * WASD: Move forward, left, back, right
 *
 */
class GLFWCameraController {

    static constexpr double MOUSE_SENS_MULTIPLIER = .001;

  public:
    GLFWCameraController(Camera camera,
                         GLFWWindowHandle window,
                         float movement_speed = 3.,
                         double mouse_sensitivity = 1.)
        : window(window), movement_speed(movement_speed), camera(camera),
          mouse_sensitivity(mouse_sensitivity) {}

    void update() {
        const double time = glfwGetTime();
        const double time_diff = time - last_time;
        last_time = time;
        if (time_diff >= 1.) {
            return;
        }

        // WASD, SHIFT, SPACE

        glm::vec3 move(0);

        if (glfwGetKey(*window, GLFW_KEY_W) == GLFW_PRESS)
            move.z += time_diff * movement_speed;
        if (glfwGetKey(*window, GLFW_KEY_S) == GLFW_PRESS)
            move.z -= time_diff * movement_speed;
        if (glfwGetKey(*window, GLFW_KEY_D) == GLFW_PRESS)
            move.x += time_diff * movement_speed;
        if (glfwGetKey(*window, GLFW_KEY_A) == GLFW_PRESS)
            move.x -= time_diff * movement_speed;
        if (glfwGetKey(*window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
            move.y -= time_diff * movement_speed;
        if (glfwGetKey(*window, GLFW_KEY_SPACE) == GLFW_PRESS)
            move.y += time_diff * movement_speed;

        if (glm::length(move) > 1e-7)
            camera.fly(move.x, move.y, -move.z);

        // MOUSE

        if (!raw_mouse_input && glfwGetMouseButton(*window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS) {
            raw_mouse_input = true;
            glfwSetInputMode(*window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            glfwSetInputMode(*window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
            glfwGetCursorPos(*window, &mouse_last_x, &mouse_last_y);
        }
        if (raw_mouse_input) {
            if (glfwGetKey(*window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                raw_mouse_input = false;
                glfwSetInputMode(*window, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
                glfwSetInputMode(*window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            } else {
                double mouse_x, mouse_y;
                glfwGetCursorPos(*window, &mouse_x, &mouse_y);
                const double mouse_x_diff = mouse_x - mouse_last_x; // right
                const double mouse_y_diff = mouse_y - mouse_last_y; // down
                mouse_last_x = mouse_x;
                mouse_last_y = mouse_y;

                glm::vec2 rot(mouse_x_diff * mouse_sensitivity * MOUSE_SENS_MULTIPLIER,
                              - mouse_y_diff * mouse_sensitivity * MOUSE_SENS_MULTIPLIER);
                if (glm::length(rot) > 1e-7)
                    camera.rotate(rot.x, rot.y);
            }
        }
    }

    const Camera& get_camera() const {
        return camera;
    }

  private:
    GLFWWindowHandle window;
    float movement_speed;
    Camera camera;
    double last_time = 0;

    double mouse_sensitivity;
    bool raw_mouse_input = false;
    double mouse_last_x = 0;
    double mouse_last_y = 0;
};

} // namespace merian
