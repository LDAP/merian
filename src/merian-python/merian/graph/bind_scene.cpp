// Python.h must precede the standard headers
#include "bindings.hpp"

#include "merian-graph/nodes/scene_file/scene_file_node.hpp"
#include "merian-shaders/shading/materials/gltf_material.hpp"
#include "merian/utils/normal_encoding.hpp"

#include <cstring>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace merian::python {

using namespace nb::literals;

namespace {

using SceneReference = NodeReference<SceneFileNode>;
using Matrix4 = nb::ndarray<const float, nb::shape<4, 4>, nb::c_contig, nb::device::cpu>;
using Points = nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Uvs = nb::ndarray<const float, nb::shape<-1, 2>, nb::c_contig, nb::device::cpu>;
using Triangles = nb::ndarray<const uint32_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Pixels = nb::ndarray<const uint8_t, nb::shape<-1, -1, 4>, nb::c_contig, nb::device::cpu>;

struct CameraReference : SceneReference {
    // the active camera where empty
    std::optional<uint32_t> index;
};

struct MeshEntry {
    Scene::MeshID mesh_id;
    std::string name;
    MaterialID material_id;
    std::vector<Scene::NodeID> instances;
};

Scene& get_scene(const SceneReference& reference) {
    reference.python_graph().ensure_editable();
    const SceneHandle& scene = reference.get()->get_scene();
    if (!scene) {
        throw std::runtime_error(fmt::format(
            "the scene of '{}' does not exist before the graph connected", reference.identifier));
    }
    return *scene;
}

Camera& get_camera(const CameraReference& reference) {
    const Scene& scene = get_scene(reference);
    const std::vector<CameraHandle> cameras = scene.get_cameras();
    if (cameras.empty()) {
        throw std::runtime_error(
            fmt::format("the scene of '{}' has no camera", reference.identifier));
    }
    if (!reference.index) {
        return *scene.get_active_camera();
    }
    if (*reference.index >= cameras.size()) {
        throw nb::index_error(fmt::format("no camera {}", *reference.index).c_str());
    }
    return *cameras[*reference.index];
}

const Scene::Node& get_node(const Scene& scene, const Scene::NodeID node_id) {
    const auto& nodes = scene.get_scene_graph();
    if (node_id >= nodes.size() || !nodes[node_id]) {
        throw nb::key_error(fmt::format("no node {}", node_id).c_str());
    }
    return *nodes[node_id];
}

const Scene::MeshInfo& get_mesh(const Scene& scene, const Scene::MeshID mesh_id) {
    const auto& infos = scene.get_mesh_infos();
    if (mesh_id >= infos.size() || !infos[mesh_id].mesh) {
        throw nb::key_error(fmt::format("no mesh {}", mesh_id).c_str());
    }
    return infos[mesh_id];
}

void check_material(const Scene& scene, const MaterialID material_id) {
    if (material_id >= scene.get_material_system()->get_material_count()) {
        throw nb::key_error(fmt::format("no material {}", material_id).c_str());
    }
}

void check_texture(const Scene& scene, const TextureID texture_id) {
    if (!scene.get_texture_manager()->has_texture(texture_id)) {
        throw nb::key_error(fmt::format("no texture {}", texture_id).c_str());
    }
}

void check_pixels(const Pixels& pixels) {
    if (pixels.shape(0) == 0 || pixels.shape(1) == 0) {
        throw std::invalid_argument("pixels is empty");
    }
}

const uint32_t* rgba(const Pixels& pixels) {
    return reinterpret_cast<const uint32_t*>(pixels.data());
}

Scene::SimpleMesh& get_morphed_mesh(const Scene& scene, const Scene::MeshID mesh_id) {
    auto* const mesh = dynamic_cast<Scene::SimpleMesh*>(get_mesh(scene, mesh_id).mesh.get());
    if (!mesh || !mesh->is_morphed()) {
        throw std::invalid_argument(
            fmt::format("mesh {} was not added with add_mesh(morphed=True)", mesh_id));
    }
    return *mesh;
}

float4x4 to_float4x4(const Matrix4& m) {
    float4x4 result;
    std::memcpy(&result[0][0], m.data(), sizeof(result));
    return result;
}

nb::ndarray<nb::numpy, float, nb::shape<4, 4>> to_array(const float4x4& m) {
    auto* const data = new float4x4(m);
    const nb::capsule owner(data,
                            [](void* pointer) noexcept { delete static_cast<float4x4*>(pointer); });
    return {&(*data)[0][0], {4, 4}, owner};
}

nb::tuple to_tuple(const float3& v) {
    return nb::make_tuple(v.x, v.y, v.z);
}

float3 to_float3(const std::array<float, 3>& v) {
    return {v[0], v[1], v[2]};
}

float3 point(const Points& points, const size_t index) {
    return {points(index, 0), points(index, 1), points(index, 2)};
}

void write_positions(Scene::SimpleMesh& mesh, const Points& positions) {
    for (size_t i = 0; i < mesh.vertices.size(); i++) {
        mesh.vertices[i].position = point(positions, i);
    }
}

void write_uvs(Scene::SimpleMesh& mesh, const Uvs& uvs) {
    if (uvs.shape(0) != mesh.vertices.size()) {
        throw std::invalid_argument("uvs and positions differ in count");
    }
    for (size_t i = 0; i < mesh.vertices.size(); i++) {
        mesh.vertices[i].uv = half2(float2(uvs(i, 0), uvs(i, 1)));
    }
}

void write_normals(Scene::SimpleMesh& mesh, const Points& normals) {
    if (normals.shape(0) != mesh.vertices.size()) {
        throw std::invalid_argument("normals and positions differ in count");
    }
    for (size_t i = 0; i < mesh.vertices.size(); i++) {
        mesh.vertices[i].encoded_normal = encode_normal(normalize(point(normals, i)));
    }
}

GltfMaterial to_material(const Scene& scene,
                         const std::array<float, 4>& base_color,
                         const std::optional<TextureID>& base_color_texture,
                         const std::array<float, 3>& emission,
                         const std::optional<TextureID>& emission_texture,
                         const float roughness,
                         const float metallic) {
    GltfMaterial material;
    material.base_color_factor = {base_color[0], base_color[1], base_color[2], base_color[3]};
    material.emissive_factor = to_float3(emission);
    material.roughness_factor = roughness;
    material.metallic_factor = metallic;
    if (base_color_texture) {
        check_texture(scene, *base_color_texture);
        material.header.alpha_texture_id = *base_color_texture;
    }
    if (emission_texture) {
        check_texture(scene, *emission_texture);
        material.emissive_texture = *emission_texture;
    }
    return material;
}

} // namespace

void bind_scene(nb::module_& m) {
    nb::class_<CameraReference>(
        m, "SceneCamera",
        "A camera of a SceneNode's scene, looked up on every access; angles in radians. A camera "
        "controller connected to the scene node moves the active camera too.",
        nb::type_slots(graph_reference_slots<CameraReference>))
        .def_prop_rw(
            "position",
            [](const CameraReference& camera) {
                return to_tuple(get_camera(camera).get_position());
            },
            [](const CameraReference& camera, const std::array<float, 3>& position) {
                get_camera(camera).set_position(to_float3(position));
            })
        .def_prop_rw(
            "target",
            [](const CameraReference& camera) { return to_tuple(get_camera(camera).get_target()); },
            [](const CameraReference& camera, const std::array<float, 3>& target) {
                get_camera(camera).set_target(to_float3(target));
            })
        .def_prop_rw(
            "up",
            [](const CameraReference& camera) { return to_tuple(get_camera(camera).get_up()); },
            [](const CameraReference& camera, const std::array<float, 3>& up) {
                get_camera(camera).set_up(to_float3(up));
            })
        .def_prop_rw(
            "fov_vertical",
            [](const CameraReference& camera) {
                return get_camera(camera).get_field_of_view_vertical();
            },
            [](const CameraReference& camera, const float fov) {
                get_camera(camera).set_field_of_view_vertical(fov);
            })
        .def_prop_rw(
            "fov_horizontal",
            [](const CameraReference& camera) {
                return get_camera(camera).get_field_of_view_horizontal();
            },
            [](const CameraReference& camera, const float fov) {
                get_camera(camera).set_field_of_view_horizontal(fov);
            })
        .def_prop_rw(
            "f_number",
            [](const CameraReference& camera) { return get_camera(camera).get_f_number(); },
            [](const CameraReference& camera, const float f_number) {
                get_camera(camera).set_f_number(f_number);
            })
        .def_prop_rw(
            "focus_distance",
            [](const CameraReference& camera) { return get_camera(camera).get_focus_distance(); },
            [](const CameraReference& camera, const float distance) {
                get_camera(camera).set_focus_distance(distance);
            })
        .def(
            "look_at",
            [](const CameraReference& camera, const std::array<float, 3>& position,
               const std::array<float, 3>& target, const std::array<float, 3>& up) {
                get_camera(camera).look_at(to_float3(position), to_float3(target), to_float3(up));
            },
            "position"_a, "target"_a, "up"_a);

    nb::class_<MeshEntry>(m, "SceneMesh", "A mesh of a SceneNode's scene when it was listed.")
        .def_ro("mesh_id", &MeshEntry::mesh_id)
        .def_ro("name", &MeshEntry::name)
        .def_ro("material_id", &MeshEntry::material_id)
        .def_prop_ro(
            "instances",
            [](const MeshEntry& mesh) {
                nb::list instances;
                for (const Scene::NodeID node_id : mesh.instances) {
                    instances.append(node_id);
                }
                return nb::tuple(instances);
            },
            "The ids of the nodes that instance the mesh.")
        .def("__repr__", [](const MeshEntry& mesh) {
            return fmt::format("SceneMesh({}, '{}')", mesh.mesh_id, mesh.name);
        });

    bind_node_reference<SceneFileNode>(
        m, "SceneNode",
        "A scene file node (PBRT, glTF, FBX), Graph.find(SceneNode, identifier); the Quake node is "
        "none. Its scene exists once the graph connected. Edits apply in the iteration that "
        "follows "
        "them, from before_iteration in the one it prepares, and, "
        "like graph edits, are allowed between runs or from the iteration callbacks of Graph.run. "
        "Loading a file discards the edits and invalidates the ids.")
        .def_prop_ro(
            "camera",
            [](const SceneReference& reference) { return CameraReference{reference, {}}; },
            "The active camera, whichever it is when accessed.")
        .def_prop_ro("cameras",
                     [](const SceneReference& reference) {
                         nb::list cameras;
                         for (uint32_t index = 0; index < get_scene(reference).get_cameras().size();
                              index++) {
                             cameras.append(CameraReference{reference, index});
                         }
                         return cameras;
                     })
        .def_prop_rw(
            "active_camera",
            [](const SceneReference& reference) {
                return get_scene(reference).get_active_camera_index();
            },
            [](const SceneReference& reference, const uint32_t index) {
                Scene& scene = get_scene(reference);
                if (index >= scene.get_cameras().size()) {
                    throw nb::index_error(fmt::format("no camera {}", index).c_str());
                }
                scene.set_active_camera(index);
            },
            "The index of the active camera in cameras.")
        .def_prop_ro(
            "meshes",
            [](const SceneReference& reference) {
                std::vector<MeshEntry> meshes;
                const std::vector<Scene::MeshInfo>& infos = get_scene(reference).get_mesh_infos();
                for (Scene::MeshID mesh_id = 0; mesh_id < infos.size(); mesh_id++) {
                    if (infos[mesh_id].mesh) {
                        meshes.push_back(
                            {mesh_id,
                             infos[mesh_id].mesh->name,
                             infos[mesh_id].mesh->material_id,
                             {infos[mesh_id].instances.begin(), infos[mesh_id].instances.end()}});
                    }
                }
                return meshes;
            },
            "list[SceneMesh]")
        .def(
            "transform",
            [](const SceneReference& reference, const Scene::NodeID node_id) {
                Scene& scene = get_scene(reference);
                get_node(scene, node_id);
                return to_array(scene.get_global_transform(node_id));
            },
            "node_id"_a, "The node's world transform, 4x4 float32 row-major.")
        .def(
            "add_node",
            [](const SceneReference& reference, const std::optional<Matrix4>& transform,
               const std::optional<Scene::NodeID>& parent, const bool animated,
               const std::string& name) {
                Scene& scene = get_scene(reference);
                Scene::Node node;
                node.name = name;
                node.is_animated = animated;
                if (parent) {
                    get_node(scene, *parent);
                    node.parent = *parent;
                }
                if (transform) {
                    node.local_transform = to_float4x4(*transform);
                }
                return scene.add_node(std::move(node));
            },
            "transform"_a = nb::none(), "parent"_a = nb::none(), "animated"_a = true, "name"_a = "",
            "The node id. Only an animated node can be moved later; a file's nodes are not, so a "
            "mesh moves by instancing it under an animated node, e.g. at its transform(). "
            "transform: 4x4 float32, row-major, relative to parent.")
        .def(
            "set_transform",
            [](const SceneReference& reference, const Scene::NodeID node_id,
               const Matrix4& transform) {
                Scene& scene = get_scene(reference);
                if (!get_node(scene, node_id).is_animated) {
                    throw std::invalid_argument(
                        fmt::format("node {} was not added as animated", node_id));
                }
                scene.update_node(node_id, to_float4x4(transform));
            },
            "node_id"_a, "transform"_a, "Relative to the node's parent.")
        .def(
            "remove_node",
            [](const SceneReference& reference, const Scene::NodeID node_id) {
                Scene& scene = get_scene(reference);
                get_node(scene, node_id);
                scene.remove_node(node_id);
            },
            "node_id"_a, "Removes the node, its children and their instances.")
        .def(
            "add_mesh_instance",
            [](const SceneReference& reference, const Scene::MeshID mesh_id,
               const Scene::NodeID node_id) {
                Scene& scene = get_scene(reference);
                get_mesh(scene, mesh_id);
                get_node(scene, node_id);
                scene.add_mesh_instance(mesh_id, node_id);
            },
            "mesh_id"_a, "node_id"_a)
        .def(
            "remove_mesh_instance",
            [](const SceneReference& reference, const Scene::MeshID mesh_id,
               const Scene::NodeID node_id) {
                Scene& scene = get_scene(reference);
                if (!get_mesh(scene, mesh_id).instances.contains(node_id)) {
                    throw nb::key_error(
                        fmt::format("node {} does not instance mesh {}", node_id, mesh_id).c_str());
                }
                scene.remove_mesh_instance(mesh_id, node_id);
            },
            "mesh_id"_a, "node_id"_a)
        .def(
            "add_mesh",
            [](const SceneReference& reference, const Points& positions, const Triangles& indices,
               const MaterialID material_id, const std::optional<Points>& normals,
               const std::optional<Uvs>& uvs, const bool morphed, const bool two_sided,
               const std::string& name) {
                Scene& scene = get_scene(reference);
                check_material(scene, material_id);
                if (positions.shape(0) == 0 || indices.shape(0) == 0) {
                    throw std::invalid_argument("the mesh has no triangles");
                }
                auto mesh = std::make_unique<Scene::SimpleMesh>();
                mesh->name = name;
                mesh->material_id = material_id;
                mesh->flags = Scene::MeshFlags::IsOpaque;
                if (morphed) {
                    mesh->flags = mesh->flags | Scene::MeshFlags::IsMorphed;
                }
                if (two_sided) {
                    mesh->flags = mesh->flags | Scene::MeshFlags::TwoSided;
                }
                mesh->indices.resize(indices.shape(0));
                for (size_t i = 0; i < indices.shape(0); i++) {
                    mesh->indices[i] = {indices(i, 0), indices(i, 1), indices(i, 2)};
                    if (any(greaterThanEqual(mesh->indices[i], uint3(positions.shape(0))))) {
                        throw std::invalid_argument(
                            fmt::format("triangle {} indexes no vertex", i));
                    }
                }
                mesh->vertices.resize(positions.shape(0));
                write_positions(*mesh, positions);
                if (normals) {
                    write_normals(*mesh, *normals);
                } else {
                    std::vector<float3> points(positions.shape(0));
                    for (size_t i = 0; i < points.size(); i++) {
                        points[i] = point(positions, i);
                    }
                    const std::vector<float3> computed =
                        compute_vertex_normals(points, mesh->indices);
                    for (size_t i = 0; i < computed.size(); i++) {
                        mesh->vertices[i].encoded_normal = encode_normal(computed[i]);
                    }
                }
                if (uvs) {
                    write_uvs(*mesh, *uvs);
                }
                return scene.add_mesh(std::move(mesh));
            },
            "positions"_a, "indices"_a, "material_id"_a, "normals"_a = nb::none(),
            "uvs"_a = nb::none(), "morphed"_a = false, "two_sided"_a = false, "name"_a = "",
            "The mesh id. positions, normals: (N, 3) float32, area-weighted normals where none "
            "are given; uvs: (N, 2) float32, 0 where none are given; indices: (M, 3) uint32; "
            "material_id: from meshes or add_material. Only a morphed mesh takes set_vertices.")
        .def(
            "remove_mesh",
            [](const SceneReference& reference, const Scene::MeshID mesh_id) {
                Scene& scene = get_scene(reference);
                get_mesh(scene, mesh_id);
                scene.remove_mesh(mesh_id);
            },
            "mesh_id"_a, "Removes the mesh and its instances.")
        .def(
            "set_vertices",
            [](const SceneReference& reference, const Scene::MeshID mesh_id,
               const Points& positions, const std::optional<Points>& normals) {
                Scene::SimpleMesh& mesh = get_morphed_mesh(get_scene(reference), mesh_id);
                if (positions.shape(0) != mesh.vertices.size()) {
                    throw std::invalid_argument("set_vertices cannot change the vertex count");
                }
                write_positions(mesh, positions);
                if (normals) {
                    write_normals(mesh, *normals);
                }
                mesh.vertices_dirty = true;
            },
            "mesh_id"_a, "positions"_a, "normals"_a = nb::none(),
            "Keeps the normals where none are given.")
        .def(
            "set_mesh_material",
            [](const SceneReference& reference, const Scene::MeshID mesh_id,
               const MaterialID material_id) {
                Scene& scene = get_scene(reference);
                const Scene::MeshInfo& info = get_mesh(scene, mesh_id);
                check_material(scene, material_id);
                info.mesh->material_id = material_id;
            },
            "mesh_id"_a, "material_id"_a)
        .def(
            "add_texture",
            [](const SceneReference& reference, const Pixels& pixels, const bool srgb,
               const bool mipmaps) {
                Scene& scene = get_scene(reference);
                check_pixels(pixels);
                return scene.get_texture_manager()->add_texture_from_rgba8(
                    rgba(pixels), pixels.shape(1), pixels.shape(0), vk::SamplerAddressMode::eRepeat,
                    vk::Filter::eLinear, vk::Filter::eLinear, srgb, mipmaps);
            },
            nb::arg("pixels").noconvert(), "srgb"_a = true, "mipmaps"_a = true,
            "The texture id. pixels: (H, W, 4) uint8 RGBA, row 0 at v = 0.")
        .def(
            "set_texture",
            [](const SceneReference& reference, const TextureID texture_id, const Pixels& pixels,
               const bool srgb, const bool mipmaps) {
                Scene& scene = get_scene(reference);
                check_texture(scene, texture_id);
                check_pixels(pixels);
                scene.get_texture_manager()->set_texture_from_rgba8(
                    texture_id, rgba(pixels), pixels.shape(1), pixels.shape(0),
                    vk::SamplerAddressMode::eRepeat, vk::Filter::eLinear, vk::Filter::eLinear, srgb,
                    mipmaps);
            },
            "texture_id"_a, nb::arg("pixels").noconvert(), "srgb"_a = true, "mipmaps"_a = true,
            "Replaces the texture and its sampling; its size may change.")
        .def(
            "remove_texture",
            [](const SceneReference& reference, const TextureID texture_id) {
                Scene& scene = get_scene(reference);
                check_texture(scene, texture_id);
                scene.get_texture_manager()->remove_texture(texture_id);
            },
            "texture_id"_a,
            "Materials that still use it sample a placeholder until add_texture reuses the id.")
        .def(
            "add_material",
            [](const SceneReference& reference, const std::array<float, 4>& base_color,
               const std::optional<TextureID>& base_color_texture,
               const std::array<float, 3>& emission,
               const std::optional<TextureID>& emission_texture, const float roughness,
               const float metallic) {
                Scene& scene = get_scene(reference);
                const GltfMaterial material =
                    to_material(scene, base_color, base_color_texture, emission, emission_texture,
                                roughness, metallic);
                const MaterialSystemHandle& materials = scene.get_material_system();
                return materials->add_material(
                    materials->register_material_type(material.variant_type_name(),
                                                      GLTF_MATERIAL_SLANG_MODULE_PATH),
                    material);
            },
            "base_color"_a = std::array<float, 4>{1, 1, 1, 1}, "base_color_texture"_a = nb::none(),
            "emission"_a = std::array<float, 3>{0, 0, 0}, "emission_texture"_a = nb::none(),
            "roughness"_a = 1.f, "metallic"_a = 0.f,
            "The material id of a glTF metallic-roughness material, defaults: white diffuse. The "
            "textures multiply base_color (linear RGBA) and emission (radiance).")
        .def(
            "set_material",
            [](const SceneReference& reference, const MaterialID material_id,
               const std::array<float, 4>& base_color,
               const std::optional<TextureID>& base_color_texture,
               const std::array<float, 3>& emission,
               const std::optional<TextureID>& emission_texture, const float roughness,
               const float metallic) {
                Scene& scene = get_scene(reference);
                check_material(scene, material_id);
                const GltfMaterial material =
                    to_material(scene, base_color, base_color_texture, emission, emission_texture,
                                roughness, metallic);
                const MaterialSystemHandle& materials = scene.get_material_system();
                if (materials->get_material_type(material_id) !=
                    materials->find_material_type(material.variant_type_name())) {
                    throw std::invalid_argument(fmt::format(
                        "material {} is not of the type add_material adds", material_id));
                }
                materials->update_material(material_id, material);
            },
            "material_id"_a, "base_color"_a = std::array<float, 4>{1, 1, 1, 1},
            "base_color_texture"_a = nb::none(), "emission"_a = std::array<float, 3>{0, 0, 0},
            "emission_texture"_a = nb::none(), "roughness"_a = 1.f, "metallic"_a = 0.f,
            "Replaces all parameters of a material of the type add_material adds.");
}

} // namespace merian::python
