A generator node for images that can be loaded with `stb_image`.
The image output is persistent since it will never change and this way the image has to be transferred only in build.

Outputs:

| Type  | Output ID | Output name | Description         | Format/Resolution                                          | Persistent |
|-------|-----------|-------------|---------------------|------------------------------------------------------------|------------|
| Image | 0         | output      | the loaded image    | `vk::Format::eR8G8B8A8Unorm` / `vk::Format::eR8G8B8A8Srgb` | yes        |
