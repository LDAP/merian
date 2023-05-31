#pragma once

#include "glm/glm.hpp"
namespace merian {

/**
 * @brief      Control a camera using keyboard and mouse.
 * 
 * The controler can use a CameraAnimator for smoothed camera motion.
 */
class CameraController {

public:

    void set_movement_speed(float movement_speed) {
        this->movement_speed = movement_speed;
    }

    float get_movement_speed() {
        return this->movement_speed;
    }

    void set_mouse_position(int x, int y);

    void set_mouse_position(glm::ivec2 mouse);

    void set_window_size(int width, int height);
    
    void set_window_size(glm::ivec2 size);


private:
    float movement_speed = 3;

};

}
