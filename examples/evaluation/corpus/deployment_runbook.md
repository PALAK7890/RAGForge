# Atlas deployment runbook
The production service is Atlas API; its health endpoint is /health/ready.
Deployments progress through canary, observation, and full rollout.
The canary receives 5% of traffic and is observed for 20 minutes.
Abort rollout if the error rate exceeds 2% for 5 consecutive minutes.
Rollback restores the previous signed release artifact; it does not roll back database migrations.
The environment variable ATLAS_REQUEST_TIMEOUT_MS defaults to 3000.
