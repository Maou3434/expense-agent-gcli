# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import os
import re

import google.auth

# Setup local authentication and environment variables
import vertexai
from google import genai
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.request_input import RequestInput
from google.adk.workflow import Workflow
from google.genai import types
from pydantic import BaseModel

from expense_agent.config import MODEL_NAME, THRESHOLD_USD

try:
    _, project_id = google.auth.default()
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
except Exception:
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "placeholder-project")
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = os.getenv(
    "GOOGLE_GENAI_USE_VERTEXAI", "False"
)

# Initialize Vertex AI to prevent auto-detection crashes during local testing
vertexai.init(project=project_id)


class ExpenseReport(BaseModel):
    """Pydantic model representing the expense report data."""

    amount: float = 0.0
    submitter: str = "Unknown"
    category: str = "Uncategorized"
    description: str = "No description provided"
    date: str = "Unknown"


def luhn_checksum(card_number: str) -> bool:
    """Validate a card number using Luhn's algorithm."""
    digits = [int(c) for c in card_number if c.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, digit in enumerate(reverse_digits):
        if i % 2 == 1:
            double = digit * 2
            checksum += double if double < 10 else double - 9
        else:
            checksum += digit
    return checksum % 10 == 0


def scrub_pii(text: str) -> tuple[str, list[str]]:
    """Scrubs SSNs and Credit Card numbers from the text."""
    redacted_categories = []

    # 1. Scrub Credit Cards
    # Find potential credit card numbers (sequences of 13-19 digits, possibly separated by spaces or hyphens)
    cc_pattern = re.compile(r'\b(?:\d[ -]*?){13,19}\b')

    def cc_replacer(match):
        val = match.group(0)
        clean_val = re.sub(r'[^0-9]', '', val)
        if luhn_checksum(clean_val):
            if "Credit Card" not in redacted_categories:
                redacted_categories.append("Credit Card")
            return "[REDACTED CREDIT CARD]"
        return val

    text = cc_pattern.sub(cc_replacer, text)

    # 2. Scrub SSNs
    # Match standard SSN formats: XXX-XX-XXXX or 9 digits (XXXXXXXXX)
    ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b')

    def ssn_replacer(match):
        if "SSN" not in redacted_categories:
            redacted_categories.append("SSN")
        return "[REDACTED SSN]"

    text = ssn_pattern.sub(ssn_replacer, text)

    return text, redacted_categories


def is_prompt_injection(text: str) -> bool:
    """Detects potential prompt injection attempts in the text."""
    text_lower = text.lower()
    injection_patterns = [
        r"ignore (?:all |previous )?(?:instructions|rules|guidelines|system|constraints|prompts)",
        r"bypass (?:the )?(?:rules|security|restrictions|review|system)",
        r"override (?:the )?(?:rules|system|instructions|guidelines)",
        r"system prompt",
        r"auto[- ]approve",
        r"force[- ]approve",
        r"force (?:the )?approval",
        r"you must approve",
        r"approve this expense",
        r"approve the expense",
        r"do not review",
        r"skip (?:the )?(?:risk|review|assessment)",
        r"ignore risk",
        r"new instruction",
        r"additional instruction",
        r"instead of reviewing"
    ]
    for pattern in injection_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def extract_expense_data(text: str) -> dict:
    """Helper function to extract expense details from a JSON event.

    Handles cases where the JSON is wrapped in a GCP Pub/Sub message
    and/or base64-encoded.
    """
    try:
        payload = json.loads(text)
    except Exception:
        return {}

    # Check for GCP Pub/Sub envelope {"message": {"data": "..."}}
    if (
        isinstance(payload, dict)
        and "message" in payload
        and isinstance(payload["message"], dict)
    ):
        payload = payload["message"]

    # Check for "data" key
    data = None
    if isinstance(payload, dict) and "data" in payload:
        data = payload["data"]

    # Decode base64 if it is a string
    if isinstance(data, str):
        try:
            decoded = base64.b64decode(data).decode("utf-8")
            data = json.loads(decoded)
        except Exception:
            pass

    # If data is a dict, use it, otherwise fall back to payload itself
    expense = data if isinstance(data, dict) else payload
    if not isinstance(expense, dict):
        expense = {}

    return expense


async def parse_input(ctx: Context, node_input: types.Content) -> Event:
    """Parses the incoming JSON event and routes the workflow based on the amount."""
    # Extract text from the input Content
    text = ""
    if hasattr(node_input, "parts") and node_input.parts:
        text = "".join(part.text for part in node_input.parts if part.text)
    elif isinstance(node_input, str):
        text = node_input

    expense_dict = extract_expense_data(text)

    # Parse into the Pydantic model
    try:
        expense = ExpenseReport(**expense_dict)
    except Exception:
        expense = ExpenseReport()

    # Store the parsed expense in the workflow context state
    ctx.state["expense"] = expense.model_dump()

    # Route based on the dollar threshold
    if expense.amount < THRESHOLD_USD:
        return Event(
            actions=EventActions(route="auto_approve"),
            output=expense.model_dump(),
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=f"Expense of ${expense.amount:.2f} by {expense.submitter} is under the threshold of ${THRESHOLD_USD:.2f}. Routing to Auto-Approve."
                    )
                ],
            ),
        )
    else:
        return Event(
            actions=EventActions(route="security_checkpoint"),
            output=expense.model_dump(),
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=f"Expense of ${expense.amount:.2f} by {expense.submitter} is equal to or over the threshold of ${THRESHOLD_USD:.2f}. Routing to Security Checkpoint."
                    )
                ],
            ),
        )


async def auto_approve(ctx: Context, node_input: dict) -> Event:
    """Auto-approves expenses that are below the threshold."""
    expense = ExpenseReport(**node_input)
    msg = (
        f"✅ **Auto-Approved**\n\n"
        f"An expense of **${expense.amount:.2f}** submitted by **{expense.submitter}** "
        f"({expense.category}: {expense.description}) on **{expense.date}** "
        f"has been automatically approved because it is under the threshold of ${THRESHOLD_USD:.2f}."
    )
    ctx.state["outcome"] = "Approved (Auto)"
    return Event(
        output={"status": "approved", "method": "auto"},
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg)]),
    )


async def security_checkpoint(ctx: Context, node_input: dict) -> Event:
    """Scrubs PII and defends against prompt injection before LLM review."""
    expense = ExpenseReport(**node_input)

    # Scrub PII
    clean_desc, redacted = scrub_pii(expense.description)
    expense.description = clean_desc

    # Update state with the scrubbed description and redacted categories
    ctx.state["expense"] = expense.model_dump()
    ctx.state["redacted_categories"] = redacted

    # Check for prompt injection on the original description
    original_desc = node_input.get("description", "")
    if is_prompt_injection(original_desc):
        ctx.state["security_event"] = True
        msg = (
            f"🚨 **Security Event Flagged**\n\n"
            f"The expense submitted by **{expense.submitter}** was flagged for a potential prompt injection attempt "
            f"in the description. It has been routed directly to human review, bypassing the LLM reviewer."
        )
        return Event(
            actions=EventActions(route="human_approval"),
            output=expense.model_dump(),
            content=types.Content(role="model", parts=[types.Part.from_text(text=msg)]),
        )

    # If clean, route to risk_review
    msg = f"🛡️ **Security Checkpoint Passed**\n\nNo prompt injection detected."
    if redacted:
        msg += f" Redacted: {', '.join(redacted)}."

    return Event(
        actions=EventActions(route="risk_review"),
        output=expense.model_dump(),
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg)]),
    )


async def risk_review(ctx: Context, node_input: dict) -> Event:
    """Uses LLM to review high-value expenses for risk factors."""
    expense = ExpenseReport(**node_input)

    prompt = (
        f"Please review the following expense report for any potential risk factors, policy violations, or anomalies:\n"
        f"- Submitter: {expense.submitter}\n"
        f"- Amount: ${expense.amount:.2f}\n"
        f"- Category: {expense.category}\n"
        f"- Description: {expense.description}\n"
        f"- Date: {expense.date}\n\n"
        f"Identify any risk factors and provide a clear, concise risk assessment and a recommendation (Approve or Reject)."
    )

    client = genai.Client()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    risk_assessment = response.text or "No assessment generated."
    ctx.state["risk_assessment"] = risk_assessment

    msg = (
        f"⚠️ **LLM Risk Assessment Generated**\n\n"
        f"**Expense Details**:\n"
        f"- Submitter: {expense.submitter}\n"
        f"- Amount: ${expense.amount:.2f}\n"
        f"- Category: {expense.category}\n"
        f"- Description: {expense.description}\n\n"
        f"**Risk Analysis**:\n"
        f"{risk_assessment}\n\n"
        f"Routing to human approval."
    )

    return Event(
        output={"expense": node_input, "risk_assessment": risk_assessment},
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg)]),
    )


async def human_approval(ctx: Context, node_input: dict):
    """Pauses the workflow for human approval and records the final outcome."""
    is_security_event = ctx.state.get("security_event", False)
    # If we don't have the user's approval input yet, pause the workflow and request it
    if not ctx.resume_inputs or "approval" not in ctx.resume_inputs:
        if is_security_event:
            message = "🚨 **SECURITY EVENT DETECTED**: This expense has been flagged for a potential prompt injection. Please review and reply with 'approve' or 'reject' to make a decision."
        else:
            message = "Please review the risk assessment and reply with 'approve' or 'reject' to make a decision."
        yield RequestInput(
            interrupt_id="approval",
            message=message,
        )
        return

    # Once resumed, read the decision from the human response
    decision = ctx.resume_inputs["approval"]
    expense_dict = ctx.state.get("expense", {})
    expense = ExpenseReport(**expense_dict)

    if decision.lower() in ["yes", "approve", "y", "approved"]:
        outcome = "Approved (Human)"
        status = "approved"
        msg = f"✅ **Expense Approved by Human**\n\nThe expense of **${expense.amount:.2f}** by **{expense.submitter}** has been approved."
    else:
        outcome = "Rejected (Human)"
        status = "rejected"
        msg = f"❌ **Expense Rejected by Human**\n\nThe expense of **${expense.amount:.2f}** by **{expense.submitter}** has been rejected."

    ctx.state["outcome"] = outcome

    yield Event(
        output={"status": status, "method": "human", "decision": decision},
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg)]),
    )


# Define the workflow graph and wire up the nodes with edges
root_agent = Workflow(
    name="expense_workflow",
    edges=[
        ("START", parse_input),
        (
            parse_input,
            {
                "auto_approve": auto_approve,
                "security_checkpoint": security_checkpoint,
            },
        ),
        (
            security_checkpoint,
            {
                "risk_review": risk_review,
                "human_approval": human_approval,
            },
        ),
        (risk_review, human_approval),
    ],
)

app = App(
    root_agent=root_agent,
    name="expense_agent",
)
