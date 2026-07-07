#!/usr/bin/env bash
# Cloud Run deploy — scales to zero, ~$0 idle for a portfolio service.
set -euo pipefail
PROJECT=data-portfolio-497401
gcloud run deploy demand-forecast \
  --source . \
  --project $PROJECT \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi --cpu 1 --max-instances 2
# smoke:
# curl "$(gcloud run services describe demand-forecast --region us-central1 --format 'value(status.url)')/forecast?h=6"
