Merian provides and uses a shader parameter system that based on [shader cursor](https://github.com/shader-slang/shader-slang.github.io/blob/74372f7df9ac691803d384d7b1586c7e18127ec3/docs/shader-cursors.md) and [parameter blocks](https://github.com/shader-slang/shader-slang.github.io/blob/74372f7df9ac691803d384d7b1586c7e18127ec3/docs/parameter-blocks.md).

TLDR:
- Features define all their parameters in a `struct` (i.e. a camera might have a matrix, an environment map a Sampler2D).
- Features that use other features compose them into their struct (i.e a scene has a camera and an environment map)
- At the top-level features are exposed as `ParameterBlock<T>` either as uniform in the entry point or globally.
- 
- The host side mirrors the logical structure of the structs and provides methods for binding based on member names.
- The ShaderField class allows setting or binding resources and recursing into subfields.
- The ParameterBlock class represents the top-level ShaderField.
