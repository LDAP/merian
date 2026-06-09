# BloomNode


| Type      | Input name | Delay |
|-----------|------------|-------|
| VkImageIn | src        | no    |

Outputs:

| Type       | Input name | Description                                 | Format/Resolution        | Persistent |
|------------|------------|---------------------------------------------|--------------------------|------------|
| VkImageOut | out        | filtered result                             | like `src`               | no         |
| VkImageOut | interm     | bright parts with vertical filter pass      | rgba16f                  | no         |
