#pragma once

#include "merian/utils/properties_json_dump.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

class UIProperties : public merian::JSONDumpProperties {
  public:
    bool is_ui() override {
        return true;
    }

    bool st_begin_child(const std::string& id,
                        const std::string& label,
                        const ChildFlags flags) override {
        labels.push_back(label);
        path.push_back(id);
        return JSONDumpProperties::st_begin_child(id, label, flags);
    }

    void st_end_child() override {
        path.pop_back();
        JSONDumpProperties::st_end_child();
    }

    void output_text(const std::string& text) override {
        texts.emplace_back(path, text);
    }

    bool shows_label(const std::string& label) const {
        return std::ranges::find(labels, label) != labels.end();
    }

    bool shows_text(const std::string& part, const std::vector<std::string>& ids = {}) const {
        return std::ranges::any_of(texts, [&](const auto& entry) {
            const auto& [text_path, text] = entry;
            auto it = text_path.begin();
            for (const std::string& id : ids) {
                it = std::find(it, text_path.end(), id);
                if (it == text_path.end()) {
                    return false;
                }
                it++;
            }
            return text.find(part) != std::string::npos;
        });
    }

  private:
    std::vector<std::string> labels;
    std::vector<std::string> path;
    std::vector<std::pair<std::vector<std::string>, std::string>> texts;
};
