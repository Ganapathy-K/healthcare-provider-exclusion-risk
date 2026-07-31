# MCP server

The two agent tools — the XGBoost risk scorer and the grounded RAG lookup over the OIG
exclusion records — are exposed over the [Model Context Protocol](https://modelcontextprotocol.io)
by `src/mcp_server.py`. Any MCP client can then discover and call them without importing this
codebase.

The server is a **wrapper, not a second implementation**: each MCP tool calls the same function
`agent.py` already calls, so the model, the RBAC and the grounding cannot drift between the agent
and the protocol.

## Tools

| Tool | Arguments | What it does |
|------|-----------|--------------|
| `score_provider_risk_tool` | `npi: str` | Exclusion-risk score + tier for one provider. A review-prioritisation signal, not a finding. |
| `query_exclusion_records_tool` | `question: str`, `role: str = "public"` | Grounded, cited answer over the exclusion records. Refuses when the records do not support an answer. |

## Access control at the protocol boundary

`query_exclusion_records_tool` defaults to `role="public"`, which retrieves **nothing** — the
same fail-closed default as `rbac.DEFAULT_ROLE`. This is verifiable: called with the default
role, the tool refuses; called as `investigator`, it answers with a cited NPI.

> ⚠️ In this demo the role is a caller-supplied argument, so a client could pass
> `role="investigator"` and read everything. That is a demonstration seam, called out in
> `mcp_server.py`. A real deployment MUST bind the role to the authenticated MCP session at the
> transport layer and ignore any role in the tool arguments.

## Run it

```bash
# from the project root, in the project venv
python src/mcp_server.py        # speaks MCP over stdio
```

## Connect Claude Desktop

Add this to `claude_desktop_config.json` (macOS:
`~/Library/Application Support/Claude/`, Windows: `%APPDATA%\Claude\`), pointing at the venv
Python and this file:

```json
{
  "mcpServers": {
    "healthcare-exclusion-risk": {
      "command": "/absolute/path/to/.venv/Scripts/python.exe",
      "args": ["/absolute/path/to/healthcare-provider-exclusion-risk/src/mcp_server.py"]
    }
  }
}
```

Restart Claude Desktop; the two tools appear under the server. A remote client instead of a
local one is a one-line change in `mcp_server.py`: `server.run(transport="streamable-http")`.
