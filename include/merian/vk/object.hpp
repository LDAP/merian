#pragma once

#include <memory>

namespace merian {

/**
 * @brief      Marker interface for Merian's obejcts.
 *
 * Obejcts often live on the device, their lifecycle is managed using a shared pointer and typically
 * must be extended after command buffer execution.
 */
class Object {};

using ObjectHandle = std::shared_ptr<Object>;
using ConstObjectHandle = std::shared_ptr<const Object>;

using UniqueObjectHandle = std::unique_ptr<Object>;

} // namespace merian
