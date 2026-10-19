#include <gtest/gtest.h>

#include "merian/utils/properties_json_load.hpp"

#include <algorithm>

using namespace merian;
using json = nlohmann::json;

namespace {

struct Values {
    bool flag = false;
    int32_t count = 0;
    float scale = 0;
    int mode = 0;
};

void read(Properties& props, Values& values) {
    props.config_bool("flag", values.flag);
    if (props.st_begin_child("child")) {
        props.config_int("count", values.count);
        props.st_end_child();
    }
    if (values.flag) {
        props.config_float("scale", values.scale);
    }
    props.config_options("mode", values.mode, {"x", "y"});
}

std::vector<std::string> sorted_unread(const JSONLoadProperties& props) {
    std::vector<std::string> unread = props.unread_keys();
    std::ranges::sort(unread);
    return unread;
}

} // namespace

TEST(JSONLoadProperties, UnreadKeys) {
    JSONLoadProperties props{json::parse(R"({"flag": true, "child": {"count": 2, "typo": 1},
                                             "scale": 1.5, "mode": "y", "unknown": 0})")};
    Values values;
    read(props, values);

    EXPECT_EQ(values.count, 2);
    EXPECT_EQ(values.scale, 1.5f);
    EXPECT_EQ(values.mode, 1);
    EXPECT_EQ(sorted_unread(props), (std::vector<std::string>{"/child/typo", "/unknown"}));
}

TEST(JSONLoadProperties, InvalidOption) {
    JSONLoadProperties props{json::parse(R"({"mode": "z"})")};
    Values values;
    read(props, values);

    EXPECT_EQ(values.mode, 0);
    EXPECT_TRUE(props.unread_keys().empty());
    EXPECT_EQ(props.invalid_values(), (std::vector<std::string>{"/mode: 'z' is none of x, y"}));
}

TEST(JSONLoadProperties, ConditionalKeyIsUnreadWithoutItsCondition) {
    JSONLoadProperties props{json::parse(R"({"scale": 1.5})")};
    Values values;
    read(props, values);

    EXPECT_EQ(values.scale, 0.f);
    EXPECT_EQ(sorted_unread(props), (std::vector<std::string>{"/scale"}));
}

TEST(JSONLoadProperties, TabsAddNoLevel) {
    JSONLoadProperties props{json::parse(R"({"count": 3})")};
    Properties& properties = props;
    int32_t count = 0;
    if (properties.st_begin_child("sections", "", Properties::ChildFlagBits::TABS)) {
        properties.config_int("count", count);
        properties.st_end_child();
    }

    EXPECT_EQ(count, 3);
    EXPECT_TRUE(props.unread_keys().empty());
}
