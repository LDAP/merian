#include <iostream>
#include <spdlog/spdlog.h>
#include <vk/context.hpp>

void setup_logging() {
#ifdef DEBUG
    spdlog::set_level(spdlog::level::debug);
#endif
}

int main(int argc, char** argv) {
    if (argc != 1) {
        std::cout << argv[0] << "takes no arguments.\n";
        return 1;
    }
    std::cout << "This is " << PROJECT_NAME << " " << VERSION << ".\n";

    setup_logging();

    Context();

    return 0;
}
