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

The report half reads real traces back: latency P50 / P95, cost per question, and citation
coverage (the share of answers that name an NPI they were shown). It uses trace.get(), not
the list endpoints, because the list endpoints return a trimmed record with no cost data.

Self-test:  python src/tracing.py
Report:     python src/tracing.py --report [limit]
"""

import os
import re
import statistics
import sys
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


# Every user-facing question enters through the router. Measuring the RAG span would
# report on half the traffic.
ROOT_SPAN_NAME = "agent"

# The RAG leg, when one ran. Used to find which NPIs the model was actually shown.
GENERATION_SPAN_NAME = "generate"

NPI_PATTERN = re.compile(r"\b\d{10}\b")

REFUSAL_MARKERS = ("can't answer", "cannot answer", "do not have", "don't have")

USD_TO_INR = 88.0


def percentile(values, fraction):
    """Nearest-rank percentile: returns an OBSERVED value, not an interpolation.

    At these sample sizes that matters -- P95 of thirty traces should be a request that
    really happened, not a number between two that did.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * len(ordered) + 0.5) - 1))
    return ordered[index]


def collect(limit=100):
    """Full records for the most recent question-level traces."""
    client = get_langfuse()
    if client is None:
        raise SystemExit("Tracing is disabled (no LANGFUSE_PUBLIC_KEY) -- nothing to report.")

    listed = client.api.trace.list(limit=limit).data
    wanted = [t for t in listed if t.name == ROOT_SPAN_NAME]
    # Re-fetched in full because the list response omits cost and output. See the docstring.
    return [client.api.trace.get(trace.id) for trace in wanted]


def generation_span(trace):
    """The RAG generation observation on this trace, or None when the router went to scoring."""
    for observation in getattr(trace, "observations", []) or []:
        if observation.name == GENERATION_SPAN_NAME:
            return observation
    return None


def answer_and_shown_npis(trace):
    """(answer text, NPIs the model was actually given) for one trace.

    The second value is the point: an answer citing an NPI that was never retrieved is a
    fabrication, and the only place that is visible is the trace.
    """
    span = generation_span(trace)
    output = getattr(span, "output", None) if span else None

    if isinstance(output, dict):
        answer = str(output.get("answer") or "")
        shown = {str(npi) for npi in (output.get("cited_npis") or [])}
        return answer, shown

    root_output = trace.output
    if isinstance(root_output, dict):
        return str(root_output.get("answer") or ""), set()
    return str(root_output or ""), set()


def summarise(traces):
    latencies = [t.latency for t in traces if t.latency is not None]
    costs = [t.total_cost for t in traces if t.total_cost]

    rag_traces = [t for t in traces if generation_span(t) is not None]

    substantive, cited, fabricated = 0, 0, 0
    for trace in rag_traces:
        answer, shown = answer_and_shown_npis(trace)
        if not answer or any(marker in answer.lower() for marker in REFUSAL_MARKERS):
            continue
        substantive += 1
        mentioned = set(NPI_PATTERN.findall(answer))
        if mentioned:
            cited += 1
        # An NPI in the answer that the trace never recorded as retrieved.
        if mentioned - shown:
            fabricated += 1

    return {
        "traces": len(traces),
        "rag_traces": len(rag_traces),
        "latency_p50": percentile(latencies, 0.50),
        "latency_p95": percentile(latencies, 0.95),
        "latency_mean": statistics.fmean(latencies) if latencies else None,
        "latency_max": max(latencies) if latencies else None,
        "priced_traces": len(costs),
        "cost_mean_usd": statistics.fmean(costs) if costs else None,
        "cost_p95_usd": percentile(costs, 0.95),
        "substantive": substantive,
        "cited": cited,
        "fabricated": fabricated,
        "citation_coverage": cited / substantive if substantive else None,
    }


def report(summary):
    def show(value, spec=".3f"):
        return "n/a" if value is None else f"{value:{spec}}"

    lines = [
        f"traces analysed        {summary['traces']}  (root span '{ROOT_SPAN_NAME}')",
        f"  of which reached RAG {summary['rag_traces']}  "
        f"(the rest were routed to the risk scorer)",
        "",
        "LATENCY, seconds",
        f"  P50                  {show(summary['latency_p50'])}",
        f"  P95                  {show(summary['latency_p95'])}",
        f"  mean                 {show(summary['latency_mean'])}",
        f"  slowest              {show(summary['latency_max'])}",
        "",
        "COST per question",
        f"  priced traces        {summary['priced_traces']} of {summary['traces']}",
    ]

    if summary["cost_mean_usd"] is not None:
        mean_inr = summary["cost_mean_usd"] * USD_TO_INR
        p95_inr = (summary["cost_p95_usd"] or 0) * USD_TO_INR
        lines += [
            f"  mean                 ${summary['cost_mean_usd']:.6f}   Rs {mean_inr:.4f}",
            f"  P95                  ${summary['cost_p95_usd']:.6f}   Rs {p95_inr:.4f}",
            f"  per 1,000 questions  ${summary['cost_mean_usd'] * 1000:.2f}   "
            f"Rs {mean_inr * 1000:.0f}",
        ]
    else:
        lines += ["  no priced traces -- token usage was not recorded at call time,",
                  "  and cost cannot be reconstructed afterwards."]

    lines += [
        "",
        "CITATION COVERAGE (RAG answers only)",
        f"  substantive answers  {summary['substantive']}  (refusals excluded)",
        f"  naming an NPI        {summary['cited']}",
        f"  coverage             {show(summary['citation_coverage'], '.1%')}",
        f"  UNRETRIEVED NPIs     {summary['fabricated']}   "
        f"<- must be 0; anything else is a fabricated identifier",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    if "--report" in sys.argv:
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        traces = collect(limit=limit)
        if not traces:
            raise SystemExit(f"No '{ROOT_SPAN_NAME}' traces found in the last {limit}. "
                             "Ask the agent a few questions first.")
        print(report(summarise(traces)))
        sys.exit(0)

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
