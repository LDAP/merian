## Device Selection

### Automatic selection

Merian attempts to select the most capable physical device by selecting devices following this priority list:
 - Dedicated GPU
 - Number of Extensions


### Manual selection

Devices that should be considered can be filtered by setting vendor ID, device ID or device name in the Context constructor. 
Alternatively, devices can be selected by defining the environment variables `MERIAN_DEFAULT_FILTER_VENDOR_ID`, `MERIAN_DEFAULT_FILTER_DEVICE_ID` or `MERIAN_DEFAULT_FILTER_DEVICE_NAME`.

Note that constructor arguments take precedence over environment variables.
