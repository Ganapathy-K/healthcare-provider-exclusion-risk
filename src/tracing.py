"""Langfuse tracing: make every answer replayable after the fact.

Ported unchanged from the insurance project -- the same three needs, and no reason to write
it twice.

Why this exists, in the shape this project's failures actually took. When someone says "it
refused a question it should have answered", the answer text alone is useless: the question
is *did retrieval miss the record, or did it retrieve the record and refuse anyway?* Those
are completely different bugs with different fixes, and this project has now had both --
the PROCTOLOGY question (retrieval missed) and the NPI-in-context bug (retrieval was fine,
the prompt demanded a citation the context could not supply). Neither is recoverable from a
screenshot. Distinguishing them by hand cost an hour on 2026-07-27; a trace records the
route, the retrieval, the prompt and the answer as one linked record.

The agent adds a second thing worth replaying: WHICH TOOL RAN. A question answered from the
wrong branch is a confident answer from the wrong half of the system, and the response text
alone rarely gives it away.

Failing open is deliberate. If Langfuse is unreachable, unconfigured, or its keys are
wrong, the app must still answer questions -- observability that can take down the thing
it observes is a liability, not a safeguard. Every function here is a no-op when
LANGFUSE_PUBLIC_KEY is unset, which is also what keeps the eval harness and unit tests
from needing credentials.
"""

import os
from contextlib import contextmanager

from dotenv import load_dotenv

# The keys live in .env like every other credential here. Reading os.environ without this
# silently reports "tracing disabled" on a correctly configured machine -- a failure mode
# that looks identical to not having set the keys at all.
load_dotenv()

TRACING_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))

_client = None


def get_langfuse():
    """The shared client, or None when tracing is switched off."""
    global _client
    if not TRACING_ENABLED:
        return None
    if _client is None:
        from langfuse import get_client
        _client = get_client()
    return _client


@contextmanager
def trace_span(name, as_type="span", **attributes):
    """Time a step and attach it to the current trace; do nothing if tracing is off.

    `as_type` is Langfuse's observation type -- "retriever", "generation" and "guardrail"
    render differently in the UI and let you filter for, say, every generation slower than
    two seconds. Using the right type is the difference between a searchable record and a
    wall of identical grey spans.

    Failures inside the tracer are swallowed; failures inside the caller's body are not.
    A network blip talking to Langfuse must never surface as a failed answer, but a bug in
    the code being traced still has to raise.
    """
    client = get_langfuse()
    if client is None:
        yield None
        return

    try:
        manager = client.start_as_current_observation(
            name=name, as_type=as_type, input=attributes or None)
    except Exception:                                  # tracer setup failed: run untraced
        yield None
        return

    with manager as span:
        yield span


def update_span(span, **fields):
    """Attach outputs/metadata to a span, tolerating a missing or broken span."""
    if span is None:
        return
    try:
        span.update(**fields)
    except Exception:
        pass


def flush():
    """Push buffered events. Needed for short-lived runs (eval batches, scripts) that
    would otherwise exit before the background sender wakes up."""
    client = get_langfuse()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        pass


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    print(f"tracing enabled : {TRACING_ENABLED}")
    print(f"host            : {os.getenv('LANGFUSE_HOST', '(default)')}")
    client = get_langfuse()
    print(f"auth check      : {client.auth_check() if client else 'n/a (disabled)'}")

    with trace_span("selftest", note="tracing.py smoke test") as span:
        update_span(span, output={"ok": True})
        sent = span is not None
    flush()
    print("sent a 'selftest' trace -- check the Langfuse dashboard" if sent
          else "nothing sent (tracing is off)")
