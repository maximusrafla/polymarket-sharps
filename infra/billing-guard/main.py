"""Auto-halt Cloud Function: disable project billing when a budget is exceeded.

Triggered by Pub/Sub messages from a Cloud Billing budget (topic `billing-halt`).
When the reported actual cost exceeds the budget amount, it unlinks the billing
account from the project, which terminates all billable resources (including the
e2-micro VM). This is the enforcement layer behind the budget's email alerts —
GCP native budgets only notify; this actually stops spend.

Re-enable later by linking the billing account back to the project (the disk and
config persist).
"""

import base64
import json
import os

import functions_framework
from googleapiclient import discovery

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
PROJECT_NAME = f"projects/{PROJECT_ID}"


@functions_framework.cloud_event
def stop_billing(cloud_event):
    pubsub_data = cloud_event.data["message"]["data"]
    notification = json.loads(base64.b64decode(pubsub_data).decode("utf-8"))

    cost = notification.get("costAmount", 0)
    budget = notification.get("budgetAmount", 0)
    print(f"budget notification: cost={cost} budget={budget}")

    if cost <= budget:
        print("under budget; no action")
        return

    billing = discovery.build("cloudbilling", "v1", cache_discovery=False)
    info = billing.projects().getBillingInfo(name=PROJECT_NAME).execute()
    if not info.get("billingEnabled"):
        print("billing already disabled; no action")
        return

    billing.projects().updateBillingInfo(
        name=PROJECT_NAME, body={"billingAccountName": ""}
    ).execute()
    print(f"BILLING DISABLED for {PROJECT_NAME} (cost {cost} > budget {budget})")
