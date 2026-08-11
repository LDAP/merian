// Parsing logic derived from pbrt-v4 (https://github.com/mmp/pbrt-v4),
// Copyright (c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys. Apache-2.0.
#include "pbrt_parser.hpp"

#include "pbrt_gzip.hpp"
#include "pbrt_spectrum.hpp"
#include "pbrt_tokenizer.hpp"

#include <spdlog/spdlog.h>

#include <charconv>
#include <set>

namespace merian::pbrt {

namespace {

using Token = Tokenizer::Token;
using Kind = Tokenizer::Token::Kind;

class Parser {
  public:
    explicit Parser(const std::filesystem::path& path)
        : desc(std::make_unique<PBRTSceneDesc>()), tokenizer(gunzip_file(path), path) {
        desc->base_dir = std::filesystem::absolute(path).parent_path();
    }

    std::unique_ptr<PBRTSceneDesc> run() {
        for (Token token = tokenizer.next(); token.kind != Kind::End; token = tokenizer.next()) {
            if (token.kind != Kind::Bare) {
                throw PBRTParseError(fmt::format("expected a directive, got '{}' at {}", token.text,
                                                 tokenizer.location()));
            }
            directive(token.text);
        }
        if (!state_stack.empty()) {
            SPDLOG_WARN("pbrt: unbalanced AttributeBegin at end of file");
        }
        return std::move(desc);
    }

  private:
    struct GraphicsState {
        float4x4 ctm = identity();
        int32_t material = ShapeDesc::MATERIAL_DEFAULT;
        std::optional<AreaLightDesc> area_light;
        int32_t inside_medium = -1;
        bool reverse_orientation = false;
    };

    float parse_number(const std::string_view text) {
        float value{};
        const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (ec != std::errc() || ptr != text.data() + text.size()) {
            throw PBRTParseError(
                fmt::format("expected a number, got '{}' at {}", text, tokenizer.location()));
        }
        return value;
    }

    // Fixed-arity numeric directives may bracket their arguments.
    float next_number() {
        for (;;) {
            const Token token = tokenizer.next();
            if (token.kind == Kind::LBracket || token.kind == Kind::RBracket) {
                continue;
            }
            if (token.kind != Kind::Bare) {
                throw PBRTParseError(fmt::format("expected a number at {}", tokenizer.location()));
            }
            return parse_number(token.text);
        }
    }

    float3 next_float3() {
        const float x = next_number();
        const float y = next_number();
        const float z = next_number();
        return float3(x, y, z);
    }

    // next_number() skips a leading '['; the matching ']' is left behind after the last argument.
    void consume_trailing_bracket() {
        if (tokenizer.peek().kind == Kind::RBracket) {
            tokenizer.next();
        }
    }

    std::string next_quoted() {
        const Token token = tokenizer.next();
        if (token.kind != Kind::String) {
            throw PBRTParseError(
                fmt::format("expected a quoted string at {}", tokenizer.location()));
        }
        return std::string(token.text);
    }

    void store_value(ParsedParameter& param, const Token& token) {
        if (token.kind == Kind::String) {
            param.strings.emplace_back(token.text);
            return;
        }
        if (token.kind != Kind::Bare) {
            throw PBRTParseError(
                fmt::format("unexpected token in parameter list at {}", tokenizer.location()));
        }
        if (token.text == "true" || token.text == "false") {
            param.bools.push_back(token.text == "true" ? 1 : 0);
            return;
        }
        if (param.type == "integer") {
            // Do not round-trip through float: mesh indices exceed its 24-bit mantissa.
            int32_t value{};
            const auto [ptr, ec] =
                std::from_chars(token.text.data(), token.text.data() + token.text.size(), value);
            if (ec != std::errc() || ptr != token.text.data() + token.text.size()) {
                value = static_cast<int32_t>(parse_number(token.text));
            }
            param.ints.push_back(value);
            param.floats.push_back(static_cast<float>(value));
            return;
        }
        param.floats.push_back(parse_number(token.text));
    }

    // Parameters are quoted "type name" declarators; a quoted value never contains a space.
    ParamDict parse_params() {
        ParamDict dict;
        while (tokenizer.peek().kind == Kind::String &&
               tokenizer.peek().text.find(' ') != std::string_view::npos) {
            const Token declarator = tokenizer.next();
            ParsedParameter param;
            param.type = std::string(declarator.text.substr(0, declarator.text.find(' ')));
            param.name = std::string(declarator.text.substr(declarator.text.find_last_of(' ') + 1));

            if (tokenizer.peek().kind == Kind::LBracket) {
                tokenizer.next();
                while (tokenizer.peek().kind != Kind::RBracket) {
                    if (tokenizer.peek().kind == Kind::End) {
                        throw PBRTParseError(
                            fmt::format("unterminated [ list at {}", tokenizer.location()));
                    }
                    store_value(param, tokenizer.next());
                }
                tokenizer.next();
            } else {
                store_value(param, tokenizer.next());
            }
            dict.params.emplace_back(std::move(param));
        }
        return dict;
    }

    void warn_once(const std::string& key, const std::string& message) {
        if (warned.insert(key).second) {
            SPDLOG_WARN("pbrt: {}", message);
        }
    }

    void apply(const float4x4& m) {
        state.ctm = mul(state.ctm, m);
    }

    // pbrt matrices are column-major in the file; merian float4x4 is row-major.
    float4x4 next_matrix() {
        float values[16];
        for (float& v : values) {
            v = next_number();
        }
        float4x4 m;
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                m[row][col] = values[4 * col + row];
            }
        }
        return m;
    }

    void look_at() {
        const float3 eye = next_float3();
        const float3 look = next_float3();
        const float3 up = next_float3();

        // pbrt camera space: +z towards the target.
        const float3 dir = normalize(look - eye);
        float3 right = cross(normalize(up), dir);
        if (length(right) < 1e-6f) {
            SPDLOG_WARN("pbrt: LookAt up is parallel to the view direction");
            right = cross(float3(0, 1, 0.1f), dir);
        }
        right = normalize(right);
        const float3 new_up = cross(dir, right);

        float4x4 camera_to_world = identity();
        for (int row = 0; row < 3; row++) {
            camera_to_world[row][0] = right[row];
            camera_to_world[row][1] = new_up[row];
            camera_to_world[row][2] = dir[row];
            camera_to_world[row][3] = eye[row];
        }
        apply(inverse(camera_to_world));
    }

    std::optional<float3> param_to_rgb(const ParsedParameter& param) {
        if (param.type == "rgb" || param.type == "color") {
            if (param.floats.size() >= 3) {
                return float3(param.floats[0], param.floats[1], param.floats[2]);
            }
            return std::nullopt;
        }
        if (param.type == "float" && !param.floats.empty()) {
            return float3(param.floats[0]);
        }
        return spectrum_param_rgb(param, desc->base_dir);
    }

    void handle_shape() {
        ShapeDesc shape;
        shape.type = next_quoted();
        shape.params = parse_params();
        shape.ctm = state.ctm;
        shape.reverse_orientation = state.reverse_orientation;
        shape.material = state.material;
        shape.area_light = state.area_light;
        shape.inside_medium = state.inside_medium;
        shape.object = current_object;
        if (current_object >= 0) {
            desc->objects[current_object].shape_indices.push_back(
                static_cast<int32_t>(desc->shapes.size()));
        }
        desc->shapes.emplace_back(std::move(shape));
    }

    void handle_material_inline() {
        const std::string type = next_quoted();
        ParamDict params = parse_params();
        if (type.empty() || type == "none" || type == "interface") {
            state.material = ShapeDesc::MATERIAL_INTERFACE;
            return;
        }
        state.material = static_cast<int32_t>(desc->materials.size());
        desc->materials.emplace_back(MaterialDesc{"", type, std::move(params)});
    }

    void handle_make_named_material() {
        const std::string name = next_quoted();
        ParamDict params = parse_params();
        const std::string type = params.get_string("type", "diffuse");
        const int32_t index = static_cast<int32_t>(desc->materials.size());
        desc->materials.emplace_back(MaterialDesc{name, type, std::move(params)});
        const auto [it, inserted] = desc->named_material_index.try_emplace(name, index);
        if (!inserted) {
            SPDLOG_WARN("pbrt: material '{}' redefined", name);
            it->second = index;
        }
    }

    void handle_named_material() {
        const std::string name = next_quoted();
        const auto it = desc->named_material_index.find(name);
        if (it == desc->named_material_index.end()) {
            SPDLOG_WARN("pbrt: unknown material '{}' at {}", name, tokenizer.location());
            state.material = ShapeDesc::MATERIAL_DEFAULT;
            return;
        }
        state.material = it->second;
    }

    void handle_texture() {
        TextureDesc tex;
        tex.name = next_quoted();
        tex.is_float = next_quoted() == "float";
        tex.cls = next_quoted();
        tex.params = parse_params();
        const auto [it, inserted] =
            desc->texture_index.try_emplace(tex.name, static_cast<int32_t>(desc->textures.size()));
        if (!inserted) {
            SPDLOG_WARN("pbrt: texture '{}' redefined, keeping the first definition", tex.name);
            return;
        }
        desc->textures.emplace_back(std::move(tex));
    }

    void handle_area_light() {
        const std::string type = next_quoted();
        const ParamDict params = parse_params();
        if (type != "diffuse") {
            warn_once("area_" + type, fmt::format("area light '{}' unsupported", type));
        }
        AreaLightDesc light;
        light.twosided = params.get_bool("twosided", false);
        float3 radiance(1);
        if (const ParsedParameter* l_param = params.find("L")) {
            radiance = param_to_rgb(*l_param).value_or(float3(1));
        }
        light.radiance = radiance * params.get_float("scale", 1.0f);
        if (params.find("filename") != nullptr || params.find("power") != nullptr) {
            warn_once("area_params", "area light 'filename'/'power' parameters are ignored");
        }
        state.area_light = light;
    }

    void handle_light_source() {
        const std::string type = next_quoted();
        const ParamDict params = parse_params();
        if (type != "infinite") {
            warn_once("light_" + type, fmt::format("light source '{}' unsupported", type));
            return;
        }
        InfiniteLightDesc light;
        light.filename = params.get_string("filename", "");
        light.scale = params.get_float("scale", 1.0f);
        if (const ParsedParameter* l_param = params.find("L")) {
            light.radiance = param_to_rgb(*l_param);
        }
        light.ctm = state.ctm;
        desc->infinite_lights.emplace_back(std::move(light));
    }

    void handle_camera() {
        const std::string type = next_quoted();
        const ParamDict params = parse_params();
        if (type != "perspective") {
            warn_once("camera_" + type,
                      fmt::format("camera '{}' unsupported, using perspective", type));
        }
        CameraDesc camera;
        camera.world_to_camera = state.ctm;
        camera.fov = params.get_float("fov", 90.0f);
        desc->camera = camera;
        named_ctms["camera"] = state.ctm;
    }

    void handle_include(const bool is_import) {
        const std::string filename = next_quoted();
        if (is_import) {
            warn_once("import", "Import treated as Include");
        }
        // pbrt resolves all paths relative to the top-level scene file.
        const std::filesystem::path path = std::filesystem::path(filename).is_absolute()
                                               ? std::filesystem::path(filename)
                                               : desc->base_dir / filename;
        try {
            tokenizer.push_include(gunzip_file(path), path);
        } catch (const std::exception& e) {
            SPDLOG_WARN("pbrt: cannot include '{}': {}", filename, e.what());
        }
    }

    void handle_object_begin() {
        const std::string name = next_quoted();
        if (current_object >= 0) {
            SPDLOG_WARN("pbrt: nested ObjectBegin at {}", tokenizer.location());
        }
        state_stack.push_back(state);
        current_object = static_cast<int32_t>(desc->objects.size());
        object_index[name] = current_object;
        desc->objects.emplace_back(ObjectDesc{name, state.ctm, {}});
    }

    void handle_object_end() {
        if (current_object < 0) {
            SPDLOG_WARN("pbrt: ObjectEnd without ObjectBegin at {}", tokenizer.location());
            return;
        }
        current_object = -1;
        if (!state_stack.empty()) {
            state = state_stack.back();
            state_stack.pop_back();
        }
    }

    void handle_object_instance() {
        const std::string name = next_quoted();
        const auto it = object_index.find(name);
        if (it == object_index.end()) {
            SPDLOG_WARN("pbrt: unknown object '{}' at {}", name, tokenizer.location());
            return;
        }
        desc->instances.emplace_back(InstanceDesc{it->second, state.ctm});
    }

    void handle_make_named_medium() {
        const std::string name = next_quoted();
        const ParamDict params = parse_params();
        const std::string type = params.get_string("type", "homogeneous");

        MediumDesc medium;
        medium.name = name;
        if (type == "homogeneous") {
            if (const ParsedParameter* sigma_a = params.find("sigma_a")) {
                medium.sigma_a = param_to_rgb(*sigma_a).value_or(medium.sigma_a);
            }
            medium.sigma_a *= params.get_float("scale", 1.0f);
            const ParsedParameter* sigma_s = params.find("sigma_s");
            const float3 s =
                sigma_s != nullptr ? param_to_rgb(*sigma_s).value_or(float3(1)) : float3(1);
            if (s.x != 0.f || s.y != 0.f || s.z != 0.f) {
                warn_once("media_scatter", "media scattering is ignored, absorption only");
            }
        } else {
            medium.sigma_a = float3(0);
            warn_once("media_" + type, fmt::format("medium '{}' unsupported", type));
        }
        const int32_t index = static_cast<int32_t>(desc->media.size());
        desc->media.emplace_back(std::move(medium));
        desc->medium_index[name] = index;
    }

    void handle_medium_interface() {
        std::string names[2];
        for (int i = 0; i < 2 && tokenizer.peek().kind == Kind::String &&
                        tokenizer.peek().text.find(' ') == std::string_view::npos;
             i++) {
            names[i] = std::string(tokenizer.next().text);
        }
        // only the interior medium maps to material absorption
        state.inside_medium = -1;
        if (!names[0].empty()) {
            const auto it = desc->medium_index.find(names[0]);
            if (it == desc->medium_index.end()) {
                SPDLOG_WARN("pbrt: unknown medium '{}'", names[0]);
            } else {
                state.inside_medium = it->second;
            }
        }
    }

    void skip_unknown_arguments() {
        for (;;) {
            const Kind kind = tokenizer.peek().kind;
            if (kind == Kind::String) {
                tokenizer.next();
            } else if (kind == Kind::LBracket) {
                tokenizer.next();
                while (tokenizer.peek().kind != Kind::RBracket &&
                       tokenizer.peek().kind != Kind::End) {
                    tokenizer.next();
                }
                tokenizer.next();
            } else {
                return;
            }
        }
    }

    void directive(const std::string_view name) {
        // 1. transforms
        if (name == "Identity") {
            state.ctm = identity();
        } else if (name == "Translate") {
            apply(translation(next_float3()));
            consume_trailing_bracket();
        } else if (name == "Scale") {
            apply(scale(next_float3()));
            consume_trailing_bracket();
        } else if (name == "Rotate") {
            const float angle = next_number();
            const float3 axis = next_float3();
            consume_trailing_bracket();
            if (length(axis) < 1e-8f) {
                SPDLOG_WARN("pbrt: Rotate with zero axis at {}", tokenizer.location());
            } else {
                apply(rotation(axis, radians(angle)));
            }
        } else if (name == "LookAt") {
            look_at();
            consume_trailing_bracket();
        } else if (name == "Transform") {
            state.ctm = next_matrix();
            consume_trailing_bracket();
        } else if (name == "ConcatTransform") {
            apply(next_matrix());
            consume_trailing_bracket();
        } else if (name == "CoordinateSystem") {
            named_ctms[next_quoted()] = state.ctm;
        } else if (name == "CoordSysTransform") {
            const std::string sys = next_quoted();
            const auto it = named_ctms.find(sys);
            if (it == named_ctms.end()) {
                SPDLOG_WARN("pbrt: unknown coordinate system '{}'", sys);
            } else {
                state.ctm = it->second;
            }
        }
        // 2. state
        else if (name == "WorldBegin") {
            state = GraphicsState{};
        } else if (name == "WorldEnd") {
            // removed in v4, accepted for compatibility
        } else if (name == "AttributeBegin" || name == "TransformBegin") {
            // Transform* was removed in v4 but converters still emit it; like pbrt-v4, treat it
            // as Attribute* (which saves a superset of the v3 transform-only state).
            if (name == "TransformBegin") {
                warn_once("transform_begin",
                          "TransformBegin/End are deprecated, treated as AttributeBegin/End");
            }
            state_stack.push_back(state);
        } else if (name == "AttributeEnd" || name == "TransformEnd") {
            if (state_stack.empty()) {
                SPDLOG_WARN("pbrt: unmatched {} at {}", name, tokenizer.location());
            } else {
                state = state_stack.back();
                state_stack.pop_back();
            }
        } else if (name == "ReverseOrientation") {
            state.reverse_orientation = !state.reverse_orientation;
        } else if (name == "ObjectBegin") {
            handle_object_begin();
        } else if (name == "ObjectEnd") {
            handle_object_end();
        } else if (name == "ObjectInstance") {
            handle_object_instance();
        } else if (name == "Include") {
            handle_include(false);
        } else if (name == "Import") {
            handle_include(true);
        }
        // 3. entities
        else if (name == "Shape") {
            handle_shape();
        } else if (name == "Material") {
            handle_material_inline();
        } else if (name == "MakeNamedMaterial") {
            handle_make_named_material();
        } else if (name == "NamedMaterial") {
            handle_named_material();
        } else if (name == "Texture") {
            handle_texture();
        } else if (name == "AreaLightSource") {
            handle_area_light();
        } else if (name == "LightSource") {
            handle_light_source();
        } else if (name == "Camera") {
            handle_camera();
        } else if (name == "Film") {
            next_quoted();
            const ParamDict params = parse_params();
            desc->film_width = params.get_int("xresolution", desc->film_width);
            desc->film_height = params.get_int("yresolution", desc->film_height);
        }
        // 4. consumed without effect
        else if (name == "Sampler" || name == "Integrator" || name == "PixelFilter" ||
                 name == "Accelerator") {
            next_quoted();
            parse_params();
        } else if (name == "Option") {
            parse_params();
        } else if (name == "ColorSpace") {
            const std::string space = next_quoted();
            if (space != "srgb") {
                warn_once("colorspace", fmt::format("color space '{}' treated as srgb", space));
            }
        } else if (name == "MakeNamedMedium") {
            handle_make_named_medium();
        } else if (name == "MediumInterface") {
            handle_medium_interface();
        } else if (name == "Attribute") {
            next_quoted();
            parse_params();
            warn_once("attribute", "scoped 'Attribute' defaults are unsupported");
        } else if (name == "ActiveTransform") {
            tokenizer.next();
            warn_once("animated_transform", "animated transforms use the start transform");
        } else if (name == "TransformTimes") {
            next_number();
            next_number();
            consume_trailing_bracket();
            warn_once("animated_transform", "animated transforms use the start transform");
        } else {
            SPDLOG_WARN("pbrt: unknown directive '{}' at {}", name, tokenizer.location());
            skip_unknown_arguments();
        }
    }

    std::unique_ptr<PBRTSceneDesc> desc;
    Tokenizer tokenizer;

    GraphicsState state;
    std::vector<GraphicsState> state_stack;
    std::unordered_map<std::string, float4x4> named_ctms;
    std::unordered_map<std::string, int32_t> object_index;
    int32_t current_object = -1;
    std::set<std::string> warned;
};

} // namespace

std::unique_ptr<PBRTSceneDesc> parse_pbrt_file(const std::filesystem::path& path) {
    Parser parser(path);
    return parser.run();
}

} // namespace merian::pbrt
