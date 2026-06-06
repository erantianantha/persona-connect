"""
packages/calendar_tools/index.py
Python version of the Cal.com API wrappers (used by FastAPI voice app).
Mirrors the TypeScript version in packages/calendar_tools/index.ts.
"""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import TypedDict

import httpx
from dotenv import load_dotenv

load_dotenv()

CAL_API_BASE  = "https://api.cal.com/v2"
KEY           = os.environ.get("CAL_API_KEY", "")
EVENT_TYPE_ID = os.environ.get("CAL_EVENT_TYPE_ID", "")


class Slot(TypedDict):
    id: str
    startTime: str
    display: str


# ──────────────────────────────────────────────
# Date range parser
# ──────────────────────────────────────────────
def _parse_date_range(date_range: str) -> tuple[str, str]:
    now = datetime.now(tz=timezone.utc)
    lc = date_range.lower().strip()

    if "today" in lc:
        start = now
        end = now + timedelta(days=1)
    elif "tomorrow" in lc:
        start = now + timedelta(days=1)
        end = now + timedelta(days=2)
    elif "next week" in lc:
        days_until_monday = (7 - now.weekday()) % 7 or 7
        start = now + timedelta(days=days_until_monday)
        end = start + timedelta(days=5)
    elif "this week" in lc:
        start = now
        end = now + timedelta(days=7 - now.weekday())
    elif "friday" in lc:
        days_until_friday = (4 - now.weekday() + 7) % 7 or 7
        start = now + timedelta(days=days_until_friday)
        end = start + timedelta(days=1)
    else:
        # Try ISO date
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                parsed = datetime.strptime(lc, fmt).replace(tzinfo=timezone.utc)
                return parsed.strftime("%Y-%m-%dT%H:%M:%SZ"), (parsed + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                continue
        # Default: next 7 days
        start = now
        end = now + timedelta(days=7)

    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_slot(start_time: str) -> str:
    try:
        dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        return dt.strftime("%a, %b %d · %I:%M %p IST")
    except Exception:
        return start_time


# ──────────────────────────────────────────────
# check_availability
# ──────────────────────────────────────────────
def check_availability(date_range: str) -> dict:
    """Returns a dict with 'slots' list and 'message' string."""
    if not KEY or not EVENT_TYPE_ID:
        # Demo fallback
        now = datetime.now(tz=timezone.utc)
        demo_slots = [
            {
                "id": f"demo-slot-{i}",
                "startTime": (now + timedelta(days=i)).isoformat(),
                "display": f"{_format_slot((now + timedelta(days=i)).isoformat())} (demo)",
            }
            for i in range(1, 4)
        ]
        return {
            "slots": demo_slots,
            "message": (
                f"Here are 3 available slots for '{date_range}' (demo mode):\n"
                + "\n".join(f"  {i+1}. {s['display']}" for i, s in enumerate(demo_slots))
                + "\nWhich works best for you?"
            ),
        }

    start, end = _parse_date_range(date_range)
    url = f"{CAL_API_BASE}/slots?eventTypeId={EVENT_TYPE_ID}&start={start}&end={end}"

    with httpx.Client(timeout=10) as client:
        resp = client.get(
            url,
            headers={
                "Authorization": f"Bearer {KEY}",
                "cal-api-version": "2024-09-04"
            }
        )

    if resp.status_code != 200:
        return {"slots": [], "message": f"Could not fetch slots: HTTP {resp.status_code} - {resp.text}"}

    data = resp.json()
    inner_data = data.get("data", [])
    
    raw_slots = []
    if isinstance(inner_data, list):
        raw_slots = inner_data
    elif isinstance(inner_data, dict):
        if "slots" in inner_data and isinstance(inner_data["slots"], list):
            raw_slots = inner_data["slots"]
        else:
            # Flatten dictionary of dates mapping to lists of slots
            for val in inner_data.values():
                if isinstance(val, list):
                    raw_slots.extend(val)
    else:
        raw_slots = []

    raw_slots = raw_slots[:5]

    slots = []
    for s in raw_slots:
        start_time = s.get("start") or s.get("startTime") or ""
        slot_id = s.get("id") or s.get("slotId") or start_time
        if start_time:
            slots.append({
                "id":        slot_id,
                "startTime": start_time,
                "display":   _format_slot(start_time),
            })

    if slots:
        message = (
            f"I found {len(slots)} available slot{'s' if len(slots) > 1 else ''} for '{date_range}':\n"
            + "\n".join(f"  {i+1}. {s['display']}" for i, s in enumerate(slots))
            + "\nWhich works best for you?"
        )
    else:
        message = f"No available slots for '{date_range}'. Please try a different time range."

    return {"slots": slots, "message": message}


# ──────────────────────────────────────────────
# create_booking
# ──────────────────────────────────────────────
def create_booking(slot_id: str, name: str, email: str) -> dict:
    """Books a confirmed interview slot. Returns success status and message."""
    email = email.replace(" ", "").strip()
    if not KEY or not EVENT_TYPE_ID:
        uid = f"demo-{int(time.time())}"
        return {
            "success": True,
            "bookingUid": uid,
            "message": f"Demo booking confirmed for {name} ({email}). Meeting ID: {uid}.",
        }

    with httpx.Client(timeout=15) as client:
        resp = client.post(
            f"{CAL_API_BASE}/bookings",
            headers={
                "Authorization":    f"Bearer {KEY}",
                "Content-Type":     "application/json",
                "cal-api-version":  "2024-08-13"
            },
            json={
                "eventTypeId": int(EVENT_TYPE_ID),
                "start":       slot_id,
                "attendee": {
                    "name":     name,
                    "email":    email,
                    "timeZone": "Asia/Kolkata",
                },
            },
        )

    res_json = resp.json()

    if resp.status_code not in (200, 201):
        return {
            "success": False,
            "error":   res_json.get("message", "Unknown error"),
            "message": f"Sorry, booking failed: {res_json.get('message', 'Please try again.')}",
        }

    booking_data = res_json.get("data", {}) if "data" in res_json else res_json
    booking_uid = booking_data.get("uid") or booking_data.get("id") or str(booking_data)

    return {
        "success":    True,
        "bookingUid": booking_uid,
        "message":    f"Booked! Confirmation sent to {email}. Meeting ID: {booking_uid}.",
    }
