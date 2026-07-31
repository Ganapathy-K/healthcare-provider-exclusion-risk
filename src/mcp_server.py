"""Expose the agent's two tools over the Model Context Protocol.

WHAT THIS IS. The LangGraph agent already owns two tools -- an XGBoost risk scorer and a
grounded RAG lookup over the exclusion records. This file publishes those SAME two functions
through MCP, so an external client (Claude Desktop, an IDE, another agent) can call them
without importing this codebase. It is a wrapper, not a second implementation: each MCP tool
below is one line that calls the function agent.py already calls, so the model, the RBAC and
the grounding cannot drift between the agent and the protocol. That is the deliberate opposite
of the duplicated-feature-list bug this project fixed elsewhere -- one behaviour, one home.

WHY MCP AT ALL. Without it, "use this risk model" means shipping Python and asking the caller
to wire up XGBoost, Qdrant and Gemini themselves. MCP turns the two tools into a typed,
self-describing interface any compliant client can discover and call -- the docstrings below
ARE the schemas the client sees. The concept is the USB-C analogy (one connector, many
devices); this is the connector actually soldered on.

⚠️ THE SECURITY SEAM, STATED PLAINLY. The RAG tool takes a `role`. Over MCP that role is a
caller-supplied argument, which means a client could simply pass role="investigator" and read
everything -- the access control would be theatre. So the default here is `public`, which
retrieves NOTHING (rbac.DEFAULT_ROLE, fail-closed), and this docstring says outright what a
real deployment must do instead: bind the role to the AUTHENTICATED MCP SESSION at the
transport layer and ignore any role in the tool arguments. The parameter is kept only so the
demonstration can show the tiers working; it is exactly the kind of control that must move out
of the caller's hands before this is more than a portfolio piece. Naming the seam is the point,
not pretending it is closed.
"""

import sys

from mcp.server.fastmcp import FastMCP

from agent import query_leie_rag, score_provider_risk

server = FastMCP("healthcare-provider-exclusion-risk")


@server.tool()
def score_provider_risk_tool(npi: str) -> str:
    """Return the model's exclusion-RISK score for one provider, given a 10-digit NPI.

    Use this when the caller wants a probability or risk tier for a SPECIFIC provider. The
    result is a prioritisation signal for human review, not a finding that the provider is or
    will be excluded. Every failure (missing NPI, bad format, unknown provider) comes back as a
    plain sentence rather than an error, because a tool result has to be something the caller
    can act on.
    """
    return score_provider_risk(npi)


@server.tool()
def query_exclusion_records_tool(question: str, role: str = "public") -> str:
    """Answer a question about the OIG exclusion RECORDS, grounded and cited to real NPIs.

    Use this for lookups, counts, and who/why/where/when questions about excluded providers --
    anything about the data rather than a single provider's risk score. The answer is grounded:
    if the retrieved records do not support an answer, the tool declines rather than guessing.

    `role` selects the access tier (investigator / analyst / auditor / public) and defaults to
    `public`, which retrieves nothing. In production this role MUST come from the authenticated
    session, not from this argument -- see the module docstring. An unknown role falls back to
    public, so a typo narrows access rather than widening it.
    """
    return query_leie_rag(question, role=role)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    # stdio is the transport local MCP clients (Claude Desktop, IDEs) speak. A streamable-HTTP
    # transport is a one-line change (transport="streamable-http") when a remote client needs it.
    server.run(transport="stdio")
