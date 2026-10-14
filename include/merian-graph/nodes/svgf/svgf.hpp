#pragma once

#include "merian-graph/nodes/svgf/filter.hpp"
#include "merian-graph/nodes/svgf/options.hpp"
#include "merian-graph/nodes/svgf/taa.hpp"
#include "merian-graph/nodes/svgf/variance_estimate.hpp"

namespace merian {

// Spatiotemporal variance-guided filtering: Schied et al. 2017, Spatiotemporal Variance-Guided
// Filtering: Real-Time Reconstruction for Path-Traced Global Illumination.
class SVGF : public Node {
  public:
    Subgraph describe_subgraph() override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    void set_filter_levels(const uint32_t level_count);

    const std::shared_ptr<SVGFVarianceEstimate> variance_estimate =
        std::make_shared<SVGFVarianceEstimate>();
    std::vector<std::shared_ptr<SVGFFilter>> filters;
    const std::shared_ptr<SVGFTAA> taa = std::make_shared<SVGFTAA>();

    int filter_levels = 0;
    SVGFOptions options;
    SVGFFilterParameters filter_parameters;
    // Undefined: keep the format of the source.
    vk::Format overwrite_format = vk::Format::eUndefined;
};

} // namespace merian
