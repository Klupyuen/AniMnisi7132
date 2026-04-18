# Team Meeting Standardization Notes

## Final architecture

### Path 1
Camera -> HTTP JSON -> Flask `/camera_event` -> dashboard

### Path 2
Sensors / ESP32 / motor / e-stop -> HTTP JSON -> Flask `/system_state` -> dashboard

## Trigger logic

1. Camera detects item
2. Camera posts to `/camera_event`
3. Sorting / sensing logic runs
4. ESP32 posts updated system state to `/system_state`

## Dashboard columns

- Type
- Object Name
- Timestamp
- Camera image
- Weights
- Bin status
- Motor status
- Emergency stop
