# Final Two-Path Data Schema

## Endpoint 1: `/camera_event`

Header:
`x-api-key: SMART_WASTE_2026`

Payload:
```json
{
  "type": "plastic",
  "object_name": "plastic bottle",
  "timestamp": "2026-03-07T15:20:00",
  "image_base64": "optional_base64_string"
}
```

Notes:
- `type` = plastic / glass / metal / paper / others
- `object_name` = specific object such as plastic bottle, metal ball, paper cup
- `image_base64` is optional but needed if you want dashboard camera view

## Endpoint 2: `/system_state`

Header:
`x-api-key: SMART_WASTE_2026`

Payload:
```json
{
  "weights": {
    "plastic": 120,
    "glass": 400,
    "metal": 250,
    "paper": 180
  },
  "bin_levels": {
    "plastic": "NOT FULL",
    "glass": "FULL",
    "metal": "NOT FULL",
    "paper": "NOT FULL"
  },
  "emergency_stop": false,
  "motor_status": "RUNNING",
  "timestamp": "2026-03-07T15:21:00"
}
```
