# AWS for the Railway backend (templates only)

Nothing here is created for you, and nothing in this repo calls AWS unless the
environment variables below are set. The backend runs fine without AWS: Groq or
the template narrative, and models from the local `MODEL_DIR`.

## What the runtime needs

[railway-runtime-policy.json](railway-runtime-policy.json) allows exactly three things:

| Statement | Action | Resource | Used by |
|---|---|---|---|
| InvokeTwoNarrativeModelsOnly | `bedrock:InvokeModel` (also covers the Converse API) | Nova Lite and gpt-oss-120b foundation models in us-east-1 | `NARRATIVE_PROVIDER=bedrock`, or Bedrock as the fallback |
| ApplyOneGuardrail | `bedrock:ApplyGuardrail` | one guardrail | `GUARDRAIL_ID`, `GUARDRAIL_VERSION` |
| ModelArtifactsUnderOnePrefix | `s3:GetObject`, `s3:PutObject` over HTTPS only | `MODEL_BUCKET/predictive-care/models/*` | `MODEL_S3_URI`, `python -m app.ml.train --upload` |

No `s3:ListBucket`, no `bedrock:*`, no streaming, no other models, no other bucket.

## Setup steps (console, by the account owner)

1. Bedrock console, us-east-1: request access to the two models and check their exact IDs. If a model is only
   offered through a cross-region inference profile, add that profile's ARN to the first statement.
2. Create one guardrail (denied topics for dosing and diagnosis, content filters) and publish a version. Put its ID
   in place of `GUARDRAIL_ID`.
3. Create the bucket (block public access on, default encryption on). If it uses a customer-managed KMS key, add
   `kms:Decrypt` and `kms:GenerateDataKey` on that one key.
4. Replace `ACCOUNT_ID`, `GUARDRAIL_ID` and `MODEL_BUCKET` in the policy, create an IAM user for Railway with only
   this policy, and create one access key. Railway cannot assume a role through OIDC, so this is a long-lived key:
   store it only in Railway's variables and rotate it.
5. Railway variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION=us-east-1`, then any of
   `BEDROCK_MODEL_ID`, `NARRATIVE_PROVIDER`, `GUARDRAIL_ID`, `GUARDRAIL_VERSION`,
   `MODEL_S3_URI=s3://MODEL_BUCKET/predictive-care/models`, `BEDROCK_DAILY_CAP`, `GUARDRAIL_DAILY_CAP`.

## Status

The Bedrock, Guardrails and S3 code paths are tested with stubs only (fake clients and botocore's Stubber). They are
implemented but unverified against a real account.
