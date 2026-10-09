"""Gives each role only the records and fields it may see, removed BEFORE Gemini gets the prompt.

1. Filter before Gemini, not "please don't mention it": a hidden field never reaches Gemini, Langfuse or the logs.
2. Analyst loses names AND NPIs: one NPPES lookup turns an NPI back into a name.
3. Unknown role = public, which sees nothing: a typo narrows access, never widens it.
4. The LEIE is public data: this shows the method, it does not claim HIPAA.

Roles: investigator (all) · analyst (no names, no NPIs) · auditor (organisations only) · public (nothing).
"""

import sys
from dataclasses import dataclass, field

from langchain_community.docstore.document import Document

# GENERAL column values that mean an organisation, not a person.
ORGANISATION_CATEGORIES = {"OTHER BUSINESS", "BUS OWNER/EXEC", "MEDICAL PRACTICE, MD",
                           "PHYSICIAN PRACTICE (", "CHIROPRACTIC PRACT", "NURSING FIRM",
                           "HC CONGLOM - PARENT", "BILLING SERVICE CO", "UNIVERSITY/COLLEGE",
                           "LOCAL GOV'T"}


@dataclass(frozen=True)
class Role:
    name: str
    can_retrieve: bool = True
    # Fields this role may see; the rest are removed before the prompt.
    visible_fields: frozenset = field(default_factory=frozenset)
    # Restricts WHICH records are searched at all, as a Qdrant filter. None means no restriction.
    entity_filter: str | None = None
    description: str = ""


ALL_FIELDS = frozenset({"NPI", "NAME", "SPECIALTY", "STATE", "EXCLTYPE", "EXCLDATE", "GENERAL"})

# Name AND NPI go: one NPPES lookup turns an NPI back into a name.
DE_IDENTIFIED = ALL_FIELDS - {"NAME", "NPI"}

ROLES = {
    "investigator": Role(
        "investigator", visible_fields=ALL_FIELDS,
        description="Full access: every record, every field."),
    "analyst": Role(
        "analyst", visible_fields=DE_IDENTIFIED,
        description="Pattern analysis, de-identified: no names and no NPIs."),
    "auditor": Role(
        "auditor", visible_fields=ALL_FIELDS, entity_filter="organisation",
        description="Organisations only; individual practitioners are not searched."),
    "public": Role(
        "public", can_retrieve=False, visible_fields=frozenset(),
        description="No retrieval. A named default beats an accidental one."),
}

DEFAULT_ROLE = "public"


def get_role(name):
    """Gives the named role; an unknown name gives public, which sees nothing."""
    return ROLES.get((name or "").strip().lower(), ROLES[DEFAULT_ROLE])


def redact_document(document, role):
    """Gives a copy with hidden fields removed and the sentence rebuilt, because the name sits in both.

    A copy, never an edit: BM25 holds the same Document objects, so an edit would strip the index.
    """
    metadata = {key: value for key, value in document.metadata.items()
                if key in role.visible_fields}
    content = document.page_content

    if "NAME" not in role.visible_fields:
        content = (f"A {metadata.get('SPECIALTY', 'provider')} in "
                   f"{metadata.get('STATE', 'an unrecorded state')} was excluded on "
                   f"{metadata.get('EXCLDATE', 'an unrecorded date')} "
                   f"for {metadata.get('EXCLTYPE', 'an unrecorded reason')} "
                   f"({metadata.get('GENERAL', 'uncategorised')}).")

    return Document(page_content=content, metadata=metadata)


def apply(documents, role):
    """Gives the role's records only, with hidden fields removed; public gets nothing."""
    if not role.can_retrieve:
        return []
    if role.entity_filter == "organisation":
        documents = [doc for doc in documents
                     if str(doc.metadata.get("GENERAL", "")).strip() in ORGANISATION_CATEGORIES]
    return [redact_document(document, role) for document in documents]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    from retrieve import retrieve

    question = "Which acupuncturists in New York were excluded?"
    print(f"Q: {question}\n")

    for name in ("investigator", "analyst", "auditor", "public", "typo-role"):
        role = get_role(name)
        documents = apply(retrieve(question, top_k=3), role)
        print(f"--- {name} -> {role.name}: {role.description}")
        if not documents:
            print("    (no records)\n")
            continue
        for document in documents[:2]:
            print(f"    {document.page_content[:96]}")
            print(f"      fields: {sorted(document.metadata)}")
        print()
