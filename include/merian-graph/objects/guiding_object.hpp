#pragma once

#include "merian-graph/objects/slot_object.hpp"
#include "merian-shaders/sampling/guiding.hpp"

namespace merian {

using GuidingObject = SlotObject<GuidingModel>;
using DistanceGuidingObject = SlotObject<DistanceGuidingModel>;
using IrradianceCacheObject = SlotObject<IrradianceCache>;

} // namespace merian
