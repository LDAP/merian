#pragma once

#include "connector_input.hpp"

#include "merian/utils/pointer.hpp"
#include <any>

namespace merian_nodes {

// Access the outputs that are connected to your inputs.
class ConnectorIOMap {
  public:
    ConnectorIOMap(
        const std::function<OutputConnectorHandle(const InputConnectorHandle&)>& output_for_input)
        : output_for_input(output_for_input) {}

    template <
        typename T,
        typename ResourceAccessType = T::resource_access_type,
        typename OutputConnectorType = T::output_connector_type,
        std::enable_if_t<
            std::is_base_of_v<TypedInputConnector<OutputConnectorType, ResourceAccessType>, T>,
            bool> = true>
    const OutputConnectorType operator[](const std::shared_ptr<T>& input_connector) const {
        return input_connector->output_connector(output_for_input(input_connector));
    }

  private:
    const std::function<OutputConnectorHandle(const InputConnectorHandle&)> output_for_input;
};

class NodeIO {
  public:
    NodeIO(const std::function<GraphResourceHandle(const InputConnectorHandle&)>&
               resource_for_input_connector,
           const std::function<GraphResourceHandle(const OutputConnectorHandle&)>&
               resource_for_output_connector,
           const std::function<std::any&()>& get_frame_data)
        : resource_for_input_connector(resource_for_input_connector),
          resource_for_output_connector(resource_for_output_connector),
          get_frame_data(get_frame_data) {}

    template <
        typename T,
        typename ResourceAccessType = T::resource_access_type,
        typename OutputConnectorType = T::output_connector_type,
        std::enable_if_t<
            std::is_base_of_v<TypedInputConnector<OutputConnectorType, ResourceAccessType>, T>,
            bool> = true>
    ResourceAccessType operator[](const std::shared_ptr<T>& input_connector) const {
        return input_connector->resource(resource_for_input_connector(input_connector));
    }

    template <typename T,
              typename ResourceAccessType = T::resource_access_type,
              std::enable_if_t<std::is_base_of_v<TypedOutputConnector<ResourceAccessType>, T>,
                               bool> = true>
    ResourceAccessType operator[](const std::shared_ptr<T>& output_connector) const {
        return output_connector->resource(resource_for_output_connector(output_connector));
    }

    // Returns a reference to the frame data as the template type.
    //
    // If no frame data exists it will constructed with the given parameters, if the cast fails the
    // frame data will be replaced with the new type.
    template <typename T, typename... Args> T& frame_data(Args&&... args) const {
        std::any& data = get_frame_data();
        try {
            return std::any_cast<T&>(data);
        } catch (const std::bad_cast& e) {
            // bit dirty: if any is empty it raises bad_cast.
            data = std::make_any<T>(std::forward<Args>(args)...);
            return std::any_cast<T&>(data);
        }
    }

  private:
    const std::function<GraphResourceHandle(const InputConnectorHandle&)>
        resource_for_input_connector;
    const std::function<GraphResourceHandle(const OutputConnectorHandle&)>
        resource_for_output_connector;
    const std::function<std::any&()> get_frame_data;
};
} // namespace merian_nodes
