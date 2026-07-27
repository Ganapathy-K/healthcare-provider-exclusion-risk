"""Role-based access on retrieval: filter the index BEFORE the model sees anything.

Access control in a RAG system has exactly one correct position, and it is upstream of the
LLM. The tempting alternative -- retrieve everything, then instruct the model not to mention
what this user may not see -- is not access control at all. It is a request. The restricted
text still enters the prompt, so it reaches the provider's servers, appears in every trace and
log, and survives in any cached context; and a model that has been told to withhold something
can be talked out of it. Filtering at the vector-search level means the data never leaves the
database, which is the only version that holds when someone asks the awkward question.

That is the same border this project's sibling draws around PII: the LLM call is the boundary,
and anything a role may not see must be gone before it.

THE ROLES, and why these fields:

  investigator   everything. The people whose job is exclusion review.
  analyst        de-identified. Aggregate and pattern questions -- "which specialties are
                 excluded most?" -- need the specialty, state and date, and do not need to
                 know it was Mikal Gohring. Names AND NPIs go: see DE_IDENTIFIED for why
                 keeping the identifier would undo the whole thing.
  auditor        organisations only. External auditors reviewing entity-level exclusions have
                 no business reading individual practitioners' records.
  public         nothing. Present deliberately, so "no role" is a decision with a name rather
                 than an accidental default that returns everything.

⚠️ WHAT THIS IS NOT. The LEIE is a PUBLIC dataset -- every record here is already downloadable
from oig.hhs.gov, so nothing below protects a secret. It is a demonstration of the mechanism
on data safe to demonstrate with, and the mechanism is what transfers to a claims or clinical
corpus where the tiers are legally mandated rather than illustrative. Saying so plainly is
better than implying this repo enforces HIPAA.
"""

import sys
from dataclasses import dataclass, field

from langchain_community.docstore.document import Document
from qdrant_client.models import FieldCondition, Filter, MatchValue

# LEIE records for organisations carry a business name; individuals carry first/last names.
# `vectorstore.build_documents` records which, so entity type is filterable without a join.
ORGANISATION_CATEGORIES = {"OTHER BUSINESS", "BUS OWNER/EXEC", "MEDICAL PRACTICE, MD",
                           "PHYSICIAN PRACTICE (", "CHIROPRACTIC PRACT", "NURSING FIRM",
                           "HC CONGLOM - PARENT", "BILLING SERVICE CO", "UNIVERSITY/COLLEGE",
                           "LOCAL GOV'T"}


@dataclass(frozen=True)
class Role:
    name: str
    can_retrieve: bool = True
    # Metadata fields this role may see. Anything absent is stripped from the record before it
    # reaches the prompt, so it cannot be quoted, traced or logged.
    visible_fields: frozenset = field(default_factory=frozenset)
    # Restricts WHICH records are searched at all, as a Qdrant filter. None means no restriction.
    entity_filter: str | None = None
    description: str = ""


ALL_FIELDS = frozenset({"NPI", "NAME", "SPECIALTY", "STATE", "EXCLTYPE", "EXCLDATE", "GENERAL"})

# ⚠️ THE NPI GOES TOO, AND THE FIRST VERSION KEPT IT. Stripping the name while leaving the
# NPI is not anonymisation: the NPI is a direct identifier, and the NPPES registry turns one
# back into a name, an address and a phone number in a single public lookup. A role that
# cannot see names but can see NPIs can see names -- with one extra step nobody is stopped
# from taking. This is the everyday version of the re-identification problem, and it is worth
# stating because the redaction LOOKED complete: the answers contained no names at all.
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
    """Unknown roles fall to `public`, which retrieves nothing.

    Failing CLOSED is the only safe direction here. A typo in a role name must not widen
    access, and the alternative -- treating unknown as unrestricted -- turns a spelling
    mistake into a data breach.
    """
    return ROLES.get((name or "").strip().lower(), ROLES[DEFAULT_ROLE])


def is_organisation(metadata):
    """Organisation records carry a business name in the GENERAL category field."""
    return str(metadata.get("GENERAL", "")).strip() in ORGANISATION_CATEGORIES


def qdrant_filter(role):
    """The role's restriction expressed as a Qdrant filter, applied during the search.

    Returned as a filter rather than applied afterwards on purpose: post-filtering retrieves
    k records and then discards some, so an auditor asking a question that matches mostly
    individuals gets a nearly empty result and no indication why. Filtering in the search
    means k records that the role may actually see.
    """
    if role.entity_filter != "organisation":
        return None
    return Filter(should=[
        FieldCondition(key="metadata.GENERAL", match=MatchValue(value=category))
        for category in sorted(ORGANISATION_CATEGORIES)
    ])


def redact_document(document, role):
    """Return a REDACTED COPY: fields this role may not see removed, text rebuilt to match.

    ⚠️ IT MUST BE A COPY, AND THE FIRST VERSION WAS NOT. Mutating the document in place
    corrupted the shared corpus: BM25 holds the very Document objects the retriever returns,
    so a single analyst query stripped the names out of the index permanently, and the next
    investigator query came back pre-redacted. Access control that quietly destroys the data
    it is protecting is worse than none -- and it presented as the redaction "working", which
    is how it survived the first test.

    Rebuilding page_content is the other half, equally easy to miss. The name is in the
    indexed SENTENCE as well as the metadata, so clearing the metadata field alone leaves the
    name sitting in the text that goes to the model. Blanking one and not the other is the
    difference between access control and the appearance of it.
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
    """Enforce the role over retrieved documents. Empty list when the role may not retrieve.

    A second, independent check that every record is one this role may see. The Qdrant filter
    in `qdrant_filter` already restricts the search, but that filter only reaches the dense
    leg -- BM25 has no index to filter -- so this is the backstop that makes the guarantee
    hold whichever retriever produced the record.
    """
    if not role.can_retrieve:
        return []
    if role.entity_filter == "organisation":
        documents = [doc for doc in documents if is_organisation(doc.metadata)]
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
