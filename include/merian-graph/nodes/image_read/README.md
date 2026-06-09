A generator node for images that can be loaded with `stb_image`.

The image output is persistent since it will never change and this way the image has to be transferred only at the first run.

### LDRImageRead

Images are output as `vk::Format::eR8G8B8A8Unorm` or `vk::Format::eR8G8B8A8Srgb` depending on whether linear is `true`.
Note that HDR images are converted to LDR using the default method of `stb_image` use the `HDRImage` node to preserve the dynamic range.


Outputs:

| Type  | Output name | Description         | Format/Resolution                                          | Persistent |
|-------|-------------|---------------------|------------------------------------------------------------|------------|
| Image | out         | the loaded image    | `vk::Format::eR8G8B8A8Unorm` / `vk::Format::eR8G8B8A8Srgb` | yes        |


### HDRImageRead

Outputs:

| Type  | Output name | Description         | Format/Resolution                                          | Persistent |
|-------|-------------|---------------------|------------------------------------------------------------|------------|
| Image | out         | the loaded image    | `vk::Format::eR32G32B32A32Sfloat`                          | yes        |
