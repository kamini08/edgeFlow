# Initial Check Module Documentation

## Overview
The Initial Check module validates whether a model can run on a target edge device without optimization.

## Usage

### CLI
```
# Check compatibility only
python edgeflowc.py config.ef --check-only

# Check with custom device specs
python edgeflowc.py config.ef --device-spec-file devices.csv

# Skip check and force optimization
python edgeflowc.py config.ef --skip-check
```

### API
```
POST /api/check
{
  "model_path": "models/model.tflite",
  "config": {
    "target_device": "raspberry_pi_4",
    "quantize": "int8"
  }
}
```

## Device Specifications

### Built-in Devices
- Raspberry Pi (3, 4, Zero)
- NVIDIA Jetson (Nano, TX2, Xavier)
- Google Coral Dev Board
- Arduino Nano 33
- ESP32

### Custom Device Specs (CSV)
```
name,ram_mb,storage_mb,max_model_size_mb,cpu_cores
custom_device,1024,4096,100,2
```

### Custom Device Specs (JSON)
```
{
  "devices": [
    {
      "name": "custom_device",
      "ram_mb": 1024,
      "storage_mb": 4096,
      "max_model_size_mb": 100
    }
  ]
}
```

