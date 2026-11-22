# Bracken Observatory: telemetry service configuration notes

The telescope telemetry collector listens on TCP port 7443. Port 7444 is reserved for a maintenance console, not the collector.

Setting OBS_LINK_TTL_SECONDS has value 900. It controls archive download-link lifetime in seconds and does not set scientific data retention.

Telemetry clients wait 12 seconds before the first retry, then 24 seconds before the second retry. No additional retries are permitted by this profile.

The telemetry readiness endpoint is /observe/ready. A successful TCP connection alone is not a readiness assertion.

The collector accepts a maximum of 80 frames in one upload batch. Larger uploads must be divided without changing individual frame checksums.

This configuration note includes public setting names only. Authentication tokens, host certificates and signing keys are deliberately absent.

| Public setting | Value | Unit |
| --- | --- | --- |
| OBS_LINK_TTL_SECONDS | 900 | seconds |
| collector listener | 7443 | TCP port |
| upload limit | 80 | frames |

