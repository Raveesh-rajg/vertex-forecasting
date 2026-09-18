#!/usr/bin/env bash
# Run from the repository root after approving a deployment budget.
set -euo pipefail
: "${GCP_PROJECT:?Set GCP_PROJECT}"
gcloud run deploy demand-forecast \
  --source . --project "$GCP_PROJECT" --region "${GCP_REGION:-us-central1}" \
  --no-allow-unauthenticated --memory 512Mi --cpu 1 \
  --min-instances 0 --max-instances 1 --concurrency 8 --timeout 60
# Build, artifact storage, requests and egress can incur charges.
# Obtain the URL with gcloud run services describe and call using an identity token.
