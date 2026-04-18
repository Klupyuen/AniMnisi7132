# Security Note

Both POST endpoints are protected by API key authentication.

Protected endpoints:
- `/camera_event`
- `/system_state`

Required header:
`x-api-key: SMART_WASTE_2026`

If the key is missing or wrong, the server returns:
- HTTP 401 Unauthorized
