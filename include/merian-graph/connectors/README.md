## Connectors

### Overview

- `managed_`: Graph allocates and manages device memory
- `vk_`: Represents a Vulkan object, includes special treatment and synchronization
- `special_`: Connector for special purpose (metadata available in describe_outputs for example)



- `_out.hpp`: The output connector (use in describe_outputs)
- `_in.hpp`: The input connector (use in describe_inputs), receives the output from a output connector.
