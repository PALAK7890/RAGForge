# Aster Robotics: canary release procedure

Feature switch ROVER_SAFE_PATH defaults to enabled in production. Its documented values are enabled and disabled, not numeric levels.

The rover gateway readiness probe is GET /ready/rover. GET /health/rover reports process liveness and must not be substituted for readiness.

A release begins with 4 rovers in the canary cohort and an observation window of 35 minutes.

Stop rollout when command rejection exceeds 1.5% for 4 consecutive minutes. A single transient sample does not satisfy this stopping condition.

Rollback restores the preceding firmware image. It does not undo calibration changes stored on the motor controller.

After rollback, retain the failed-image identifier and observation logs. The release engineer records the result before scheduling a new canary.

