"""Offers the agent's two tools over MCP, so any MCP client can call them without this code.

1. One-line wrappers around the functions agent.py calls: the agent and MCP cannot drift apart.
2. The tool docstrings below are what the MCP client reads to pick a tool.
3. role defaults to public, which sees nothing; a real deployment takes role from the logged-in session.
"""

import sys

from mcp.server.fastmcp import FastMCP

from agent import query_leie_rag, score_provider_risk

server = FastMCP("healthcare-provider-exclusion-risk")


@server.tool()
def score_provider_risk_tool(npi: str) -> str:
    """Gives one provider's exclusion risk score from a 10-digit NPI, for review, not a finding; failures come back as a sentence."""
    return score_provider_risk(npi)[0]


@server.tool()
def query_exclusion_records_tool(question: str, role: str = "public") -> str:
    """Answers a question about the OIG exclusion records, cited to NPIs, or declines.

    role = investigator / analyst / auditor / public (default, sees nothing); an unknown role falls back to public.
    """
    return query_leie_rag(question, role=role)[0]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    # stdio = how local MCP clients talk.
    server.run(transport="stdio")
