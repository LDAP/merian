#pragma once

#include <nanobind/nanobind.h>
#include <nlohmann/json.hpp>

namespace nanobind::detail {

template <> struct type_caster<nlohmann::json> {
    NB_TYPE_CASTER(nlohmann::json, const_name("typing.Any"))

    bool from_python(handle src, uint8_t /*flags*/, cleanup_list* /*cleanup*/) noexcept {
        try {
            value = to_json(src);
            return true;
        } catch (...) {
            PyErr_Clear();
            return false;
        }
    }

    static handle
    from_cpp(const nlohmann::json& json, rv_policy /*policy*/, cleanup_list* /*cleanup*/) noexcept {
        try {
            return from_json(json).release();
        } catch (python_error& e) {
            e.restore();
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        }
        return {};
    }

    static nlohmann::json to_json(const handle src) {
        if (src.is_none()) {
            return nullptr;
        }
        if (PyBool_Check(src.ptr()) != 0) {
            return src.ptr() == Py_True;
        }
        if (PyLong_Check(src.ptr()) != 0) {
            int overflow = 0;
            const long long value = PyLong_AsLongLongAndOverflow(src.ptr(), &overflow);
            if (overflow == 0) {
                return value;
            }
            const unsigned long long unsigned_value = PyLong_AsUnsignedLongLong(src.ptr());
            if (PyErr_Occurred() != nullptr) {
                throw python_error();
            }
            return unsigned_value;
        }
        if (PyFloat_Check(src.ptr()) != 0) {
            return PyFloat_AsDouble(src.ptr());
        }
        if (PyUnicode_Check(src.ptr()) != 0) {
            return cast<std::string>(src);
        }
        if (PyDict_Check(src.ptr()) != 0) {
            nlohmann::json object = nlohmann::json::object();
            for (const auto [key, value] : borrow<dict>(src)) {
                object[cast<std::string>(key)] = to_json(value);
            }
            return object;
        }
        if (PyList_Check(src.ptr()) != 0 || PyTuple_Check(src.ptr()) != 0) {
            nlohmann::json array = nlohmann::json::array();
            for (const handle item : src) {
                array.push_back(to_json(item));
            }
            return array;
        }
        if (hasattr(src, "__fspath__")) {
            return cast<std::string>(src.attr("__fspath__")());
        }
        if (hasattr(src, "tolist")) {
            const object converted = src.attr("tolist")();
            if (!converted.type().is(src.type())) {
                return to_json(converted);
            }
        }
        throw type_error("not convertible to JSON");
    }

    static object from_json(const nlohmann::json& json) {
        switch (json.type()) {
        case nlohmann::json::value_t::null:
            return none();
        case nlohmann::json::value_t::boolean:
            return bool_(json.get<bool>());
        case nlohmann::json::value_t::number_integer:
            return int_(json.get<int64_t>());
        case nlohmann::json::value_t::number_unsigned:
            return int_(json.get<uint64_t>());
        case nlohmann::json::value_t::number_float:
            return float_(json.get<double>());
        case nlohmann::json::value_t::string: {
            const std::string& string = json.get_ref<const std::string&>();
            return str(string.data(), string.size());
        }
        case nlohmann::json::value_t::array: {
            list array;
            for (const auto& item : json) {
                array.append(from_json(item));
            }
            return array;
        }
        case nlohmann::json::value_t::object: {
            dict object;
            for (const auto& [key, item] : json.items()) {
                object[str(key.data(), key.size())] = from_json(item);
            }
            return object;
        }
        case nlohmann::json::value_t::binary:
            return bytes(json.get_binary().data(), json.get_binary().size());
        case nlohmann::json::value_t::discarded:
            break;
        }
        return none();
    }
};

} // namespace nanobind::detail
