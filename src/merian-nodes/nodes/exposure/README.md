# Autoexposure

Follows the idea from Lagarde et al. *Moving Frostbite to Physically Based Rendering 2.0* (2014).


| Type      | Input name | Delay |
|-----------|------------|-------|
| VkImageIn | src        | no    |

Outputs:

| Type        | Input name    | Description                                 | Format/Resolution         | Persistent |
|-------------|---------------|---------------------------------------------|---------------------------|------------|
| VkImageOut  | out           | exposed result                              | like `src`                | no         |
| VkBufferOut | histogram     | luminance histogram                         | internal                  | no         |
| VkBufferOut | avg_luminance | average luminance (for temporal adaption)   | a single float            | yes        |

