"use client";

/**
 * apps/chat/app/page.tsx
 * AI Persona Chat Interface — streaming chat UI + voice call
 */

import { useChat } from "ai/react";
import { useEffect, useRef, useState } from "react";

// ── Type helpers ───────────────────────────────────────────────
interface Slot { id: string; display: string }
interface AvailabilityResult { slots: Slot[]; message: string }
interface BookingResult { success: boolean; message: string; bookingUid?: string }

// ── Booking form component ─────────────────────────────────────
function BookingForm({
  slot,
  onConfirm,
  onCancel,
}: {
  slot: Slot;
  onConfirm: (name: string, email: string) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");

  return (
    <div className="booking-form">
      <p className="booking-slot">📅 {slot.display}</p>
      <input
        type="text"
        placeholder="Your full name"
        value={name}
        onChange={(e) => setName(e.target.value)}
        className="booking-input"
      />
      <input
        type="email"
        placeholder="Your email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        className="booking-input"
      />
      <div className="booking-actions">
        <button
          className="btn-confirm"
          onClick={() => name && email && onConfirm(name, email)}
          disabled={!name || !email}
        >
          Confirm Booking
        </button>
        <button className="btn-cancel" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  );
}



// ── Main chat page ─────────────────────────────────────────────
export default function ChatPage() {
  const [countdown, setCountdown] = useState<number | null>(null);

  const { messages, input, handleInputChange, handleSubmit, isLoading, append, error, reload } =
    useChat({
      api: "/api/chat",
      fetch: async (input, init) => {
        const res = await fetch(input, init);
        if (!res.ok) {
          let errMsg = `Request failed with status ${res.status}`;
          let retrySecs = null;
          try {
            const data = await res.json();
            if (data && data.error) {
              errMsg = data.error;
            }
            if (data && typeof data.retryAfter === "number") {
              retrySecs = data.retryAfter;
            }
          } catch (e) {
            // ignore JSON parse errors
          }

          if (res.status === 429 && retrySecs) {
            setCountdown(retrySecs);
          }
          throw new Error(errMsg);
        }
        return res;
      }
    });

  // Countdown timer effect
  useEffect(() => {
    if (countdown === null) return;
    if (countdown <= 0) {
      setCountdown(null);
      return;
    }
    const timer = setTimeout(() => {
      setCountdown(countdown - 1);
    }, 1000);
    return () => clearTimeout(timer);
  }, [countdown]);

  const bottomRef = useRef<HTMLDivElement>(null);
  const [pendingSlot, setPendingSlot] = useState<Slot | null>(null);
  const [bookingDone, setBookingDone] = useState(false);

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── Slot selection ─────────────────────────────────────────
  function selectSlot(slot: Slot) {
    setPendingSlot(slot);
  }

  // ── Booking confirmation ───────────────────────────────────
  async function confirmBooking(name: string, email: string) {
    if (!pendingSlot) return;
    const slot = pendingSlot; // capture before clearing
    setPendingSlot(null);
    setBookingDone(true);
    await append({
      role: "user",
      content: `Please book slot "${slot.display}" (ID: ${slot.id}) for ${name} (${email}).`,
    });
  }

  // ── Message renderer ───────────────────────────────────────
  // FIX: AI SDK v3 exposes tool results in msg.toolInvocations,
  // NOT in msg.content as an array of parts.
  function renderMessage(msg: (typeof messages)[0]) {
    const isUser = msg.role === "user";

    const textContent = typeof msg.content === "string" ? msg.content : "";

    let availResult: AvailabilityResult | null = null;
    let bookingResult: BookingResult | null = null;

    if (msg.toolInvocations) {
      for (const inv of msg.toolInvocations) {
        if (inv.state === "result") {
          const result = inv.result as Record<string, unknown>;
          if (result && "slots" in result) {
            availResult = result as unknown as AvailabilityResult;
          }
          if (result && "success" in result) {
            bookingResult = result as unknown as BookingResult;
          }
        }
      }
    }

    return (
      <div key={msg.id} className={`msg-row ${isUser ? "user" : "assistant"}`}>
        {!isUser && (
          <div className="avatar">
            <span>AD</span>
          </div>
        )}
        <div className={`bubble ${isUser ? "bubble-user" : "bubble-assistant"}`}>
          {textContent && <p className="bubble-text">{textContent}</p>}

          {/* Availability slots UI */}
          {availResult && availResult.slots.length > 0 && !bookingDone && (
            <div className="slots-container">
              <p className="slots-label">Available slots:</p>
              {availResult.slots.map((slot) => (
                <button
                  key={slot.id}
                  className="slot-btn"
                  onClick={() => selectSlot(slot)}
                >
                  📅 {slot.display}
                </button>
              ))}
            </div>
          )}

          {/* Booking success */}
          {bookingResult && bookingResult.success && (
            <div className="booking-success">
              ✅ {bookingResult.message}
            </div>
          )}
          {bookingResult && !bookingResult.success && (
            <div className="booking-error">
              ❌ {bookingResult.message}
            </div>
          )}
        </div>
        {isUser && (
          <div className="avatar user-avatar">
            <span>You</span>
          </div>
        )}
      </div>
    );
  }

  return (
    <>
      <div className="chat-shell">
        {/* Header */}
        <div className="chat-header">
          <div className="header-avatar">AD</div>
          <div className="header-info">
            <h1>Anantha Datta Eranti</h1>
            <p>CS Student · Full-Stack &amp; AI Developer · Scaler School of Technology</p>
          </div>
          <div className="status-dot" title="Online" />
        </div>

        {/* Messages */}
        <div className="messages">
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-icon">👋</div>
              <h2>Hey, I&apos;m Anantha!</h2>
              <p>
                CS student at Scaler, building AI systems and full-stack apps.
                Ask me anything — my projects, my stack, or let&apos;s find a time to chat.
              </p>

            </div>
          ) : (
            messages.map(renderMessage)
          )}

          {isLoading && (
            <div className="msg-row assistant">
              <div className="avatar"><span>AD</span></div>
              <div className="bubble bubble-assistant">
                <div className="typing">
                  <span /><span /><span />
                </div>
              </div>
            </div>
          )}

          {error && !isLoading && (
            <div className="msg-row assistant">
              <div className="avatar"><span>AD</span></div>
              <div className="bubble bubble-assistant error-bubble">
                <p className="error-title">⚠️ Something went wrong</p>
                <p className="error-detail">
                  {countdown !== null
                    ? `Rate limit reached. Please wait ${countdown} seconds before retrying.`
                    : error.message || "The request failed. Please try again."
                  }
                </p>
                <button
                  className="retry-btn"
                  onClick={() => {
                    if (countdown === null) reload();
                  }}
                  disabled={countdown !== null}
                >
                  {countdown !== null ? `⏳ Retry in ${countdown}s` : "↺ Retry"}
                </button>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Booking form (shown inline when slot selected) */}
        {pendingSlot && (
          <BookingForm
            slot={pendingSlot}
            onConfirm={confirmBooking}
            onCancel={() => setPendingSlot(null)}
          />
        )}

        {/* Input area */}
        <div className="input-area">
          <form className="input-form" onSubmit={handleSubmit}>
            <textarea
              className="input-field"
              placeholder={countdown !== null ? `Rate limited — please wait ${countdown}s…` : "Ask me anything — projects, skills, or schedule a chat…"}
              value={input}
              onChange={handleInputChange}
              rows={1}
              disabled={isLoading || countdown !== null}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  if (countdown === null && !isLoading) {
                    handleSubmit(e as unknown as React.FormEvent);
                  }
                }
              }}
            />
            <button
              type="submit"
              className="send-btn"
              disabled={isLoading || !input.trim() || countdown !== null}
              aria-label="Send"
            >
              ↑
            </button>
          </form>
          <p className="input-hint">
            Chat directly with Anantha&apos;s AI twin · Powered by RAG
          </p>
        </div>
      </div>
    </>
  );
}
