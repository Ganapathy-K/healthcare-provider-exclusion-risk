"""The columns the risk scorer reads from a provider record.

In its own module because both `ingest` (which writes the slim lookup parquet) and `agent`
(which reads it) need the list, and importing `agent` from `ingest` would pull in langgraph,
the Gemini SDK and the embedding stack just to write a file.
"""

LOOKUP_COLUMNS = [
    "NPI", "Entity Type Code", "Provider Business Mailing Address Telephone Number",
    "Provider Enumeration Date", "Last Update Date", "Provider Sex Code",
    "Healthcare Provider Primary Taxonomy Switch_1", "Is Sole Proprietor",
    "Healthcare Provider Taxonomy Code_1", "Provider Business Mailing Address State Name",
    "Provider Business Practice Location Address State Name",
    "Provider License Number State Code_1",
]
