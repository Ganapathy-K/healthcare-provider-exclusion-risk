"""Report what a question actually costs and how long it actually takes, from real traces.

Ported from the insurance project. Three numbers a demo cannot answer:

  latency P50 / P95   the median is what a demo shows; P95 is what a user complains about.
  cost per question   Langfuse prices the recorded token counts. Not reconstructable later:
                      if the counts were not sent at call time, the cost is gone.
  citation coverage   the fraction of substantive answers that name an NPI they were given.

TWO DIFFERENCES FROM THE INSURANCE VERSION, and both come from what this system is.

1. CITATIONS ARE NPIs, NOT PAGES. There is no document to point at -- the evidence is a
   provider record. So coverage is measured against the retrieved NPIs the trace recorded,
   not against a "page N" pattern in the text. An answer that names an NPI it was never
   shown is a worse failure than one that cites nothing, and only the trace can tell the
   difference, because the answer text alone looks equally confident either way.

2. THE ROOT SPAN IS 'agent', NOT THE RAG CALL. Every user-facing question enters through the
   router, which may end at the XGBoost scorer instead of retrieval. Measuring the RAG span
   would silently report on half the traffic and call it the system's latency.

⚠️ USE trace.get(), NOT observations.get_many(). The list endpoints return a TRIMMED
projection -- model, usage_details, cost and output all come back None on traces that plainly
contain them. Reading the list endpoint shows a correctly instrumented app as having no cost
data at all: a measurement failure wearing the costume of a code failure. Only the per-trace
fetch returns the full record.

Run:  python src/trace_metrics.py [limit]
"""

import re
import statistics
import sys

from tracing import get_langfuse

# Every user-facing question enters through the router. See difference 2 above.
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


def is_refusal(answer):
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def summarise(traces):
    latencies = [t.latency for t in traces if t.latency is not None]
    costs = [t.total_cost for t in traces if t.total_cost]

    rag_traces = [t for t in traces if generation_span(t) is not None]

    substantive, cited, fabricated = 0, 0, 0
    for trace in rag_traces:
        answer, shown = answer_and_shown_npis(trace)
        if not answer or is_refusal(answer):
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

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    traces = collect(limit=limit)
    if not traces:
        raise SystemExit(f"No '{ROOT_SPAN_NAME}' traces found in the last {limit}. "
                         "Ask the agent a few questions first.")
    print(report(summarise(traces)))
