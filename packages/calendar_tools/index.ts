/**
 * packages/calendar_tools/index.ts
 *
 * Cal.com API wrappers used by the Next.js chat route.
 * Python equivalent lives at packages/calendar_tools/index.py (used by FastAPI voice app).
 *
 * Tools:
 *   checkAvailability(dateRange: string) → { slots: Slot[] }
 *   createBooking({ slotId, name, email }) → { success: boolean; message: string; bookingUid: string }
 */

import { addDays, format, parseISO, startOfDay, endOfDay } from "date-fns";

const CAL_API_BASE = "https://api.cal.com/v2";
const KEY = process.env.CAL_API_KEY!;
const EVENT_TYPE_ID = process.env.CAL_EVENT_TYPE_ID!;

export interface Slot {
  id: string;
  startTime: string;
  endTime: string;
  display: string; // Human-readable: "Mon, Jun 09 · 10:00 AM"
}

export interface AvailabilityResult {
  slots: Slot[];
  dateRange: string;
  message: string;
}

export interface BookingResult {
  success: boolean;
  message: string;
  bookingUid?: string;
  error?: string;
}

// ── Date range parser ────────────────────────────────────────
function parseDateRange(dateRange: string): { start: string; end: string } {
  const now = new Date();
  const lc = dateRange.toLowerCase().trim();

  let startDate = now;
  let endDate = addDays(now, 7);

  if (lc.includes("today")) {
    startDate = now;
    endDate = addDays(now, 1);
  } else if (lc.includes("tomorrow")) {
    startDate = addDays(now, 1);
    endDate = addDays(now, 2);
  } else if (lc.includes("this week")) {
    startDate = now;
    endDate = addDays(now, 7 - now.getDay());
  } else if (lc.includes("next week")) {
    const daysUntilMonday = (8 - now.getDay()) % 7 || 7;
    startDate = addDays(now, daysUntilMonday);
    endDate = addDays(startDate, 5);
  } else if (lc.includes("this friday") || lc.includes("friday")) {
    const daysUntilFriday = (5 - now.getDay() + 7) % 7 || 7;
    startDate = addDays(now, daysUntilFriday);
    endDate = addDays(startDate, 1);
  } else {
    // Try to parse ISO date string like "2024-12-20"
    try {
      const parsed = parseISO(lc);
      if (!isNaN(parsed.getTime())) {
        startDate = startOfDay(parsed);
        endDate = endOfDay(parsed);
      }
    } catch {
      // fallback: next 7 days
    }
  }

  return {
    start: startDate.toISOString(),
    end: endDate.toISOString(),
  };
}

// ── Format slot for display ──────────────────────────────────
function formatSlot(startTime: string): string {
  const d = new Date(startTime);
  return d.toLocaleString("en-IN", {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "Asia/Kolkata",
  });
}

// ── checkAvailability ────────────────────────────────────────
export async function checkAvailability(
  dateRange: string
): Promise<AvailabilityResult> {
  if (!KEY || !EVENT_TYPE_ID) {
    // Graceful demo fallback (useful during local dev without real keys)
    const demoSlots: Slot[] = [
      {
        id: "demo-slot-1",
        startTime: addDays(new Date(), 1).toISOString(),
        endTime: addDays(new Date(), 1).toISOString(),
        display: `${formatSlot(addDays(new Date(), 1).toISOString())} (demo)`,
      },
      {
        id: "demo-slot-2",
        startTime: addDays(new Date(), 2).toISOString(),
        endTime: addDays(new Date(), 2).toISOString(),
        display: `${formatSlot(addDays(new Date(), 2).toISOString())} (demo)`,
      },
    ];
    return {
      slots: demoSlots,
      dateRange,
      message: `Here are 2 available slots for "${dateRange}" (demo mode — real slots need CAL_API_KEY).`,
    };
  }

  const { start, end } = parseDateRange(dateRange);

  const url = new URL(`${CAL_API_BASE}/slots`);
  url.searchParams.set("eventTypeId", EVENT_TYPE_ID);
  url.searchParams.set("start", start);
  url.searchParams.set("end", end);

  const res = await fetch(url.toString(), {
    headers: {
      Authorization: `Bearer ${KEY}`,
      "cal-api-version": "2024-09-04"
    },
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Cal.com API error: ${res.status} ${res.statusText} - ${errorText}`);
  }

  const data = await res.json();
  let rawSlots: any[] = [];
  if (data.data) {
    if (Array.isArray(data.data)) {
      rawSlots = data.data;
    } else if (data.data.slots && Array.isArray(data.data.slots)) {
      rawSlots = data.data.slots;
    } else if (typeof data.data === "object" && data.data !== null) {
      // Flatten dictionary of dates mapping to lists of slots
      for (const val of Object.values(data.data)) {
        if (Array.isArray(val)) {
          rawSlots.push(...val);
        }
      }
    }
  } else if (data.slots) {
    if (Array.isArray(data.slots)) {
      rawSlots = data.slots;
    }
  }

  const slots: Slot[] = rawSlots.slice(0, 5).map((s: any) => {
    const startTime = s.start ?? s.startTime;
    return {
      id: s.id ?? s.slotId ?? String(startTime),
      startTime: startTime,
      endTime: s.end ?? s.endTime ?? startTime,
      display: formatSlot(startTime),
    };
  });

  const message =
    slots.length > 0
      ? `I found ${slots.length} available slot${slots.length > 1 ? "s" : ""} for "${dateRange}": ${slots.map((s) => s.display).join(", ")}. Which works best for you?`
      : `No available slots found for "${dateRange}". Try a different time range.`;

  return { slots, dateRange, message };
}

// ── createBooking ────────────────────────────────────────────
export async function createBooking({
  slotId,
  name,
  email,
}: {
  slotId: string;
  name: string;
  email: string;
}): Promise<BookingResult> {
  email = email.replace(/\s+/g, "").trim();
  if (!KEY || !EVENT_TYPE_ID) {
    return {
      success: true,
      message: `Demo booking confirmed for ${name} (${email}). Meeting ID: demo-${Date.now()}.`,
      bookingUid: `demo-${Date.now()}`,
    };
  }

  const res = await fetch(`${CAL_API_BASE}/bookings`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${KEY}`,
      "Content-Type": "application/json",
      "cal-api-version": "2024-08-13"
    },
    body: JSON.stringify({
      eventTypeId: Number(EVENT_TYPE_ID),
      start: slotId,
      attendee: {
        name,
        email,
        timeZone: "Asia/Kolkata",
      },
    }),
  });

  const res_json = await res.json();

  if (!res.ok) {
    return {
      success: false,
      error: res_json.message ?? "Booking failed",
      message: `Sorry, I couldn't complete the booking. ${res_json.message ?? "Please try again."}`,
    };
  }

  const bookingData = res_json.data ?? res_json;
  const bookingUid = bookingData.uid ?? bookingData.id ?? String(bookingData);

  return {
    success: true,
    bookingUid,
    message: `Booked! A confirmation email has been sent to ${email}. Meeting ID: ${bookingUid}.`,
  };
}
