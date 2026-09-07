# nb4 — RAG. The answer to give, verbatim.

Pinned 2026-08-31. `04_rag_pipeline.ipynb` — **RAG, Retrieval-Augmented Generation.**
Sister file: `nb05_router_answer.md`. Read this; do not rebuild it from memory.

## The four steps — he can recite these
1. Turn each banned record into one plain sentence
2. Store those sentences in a vector database so they can be searched by meaning
3. Take a question and find the sentences closest in meaning — **semantic search**
4. Answer using only those sentences, or refuse if they don't cover it

## Step 1 — the exact template (`vectorstore.to_sentence`)
> *"{FIRSTNAME} {LASTNAME} is a {SPECIALTY} in {STATE} who was excluded on {EXCLDATE} for {EXCLTYPE} ({GENERAL})."*

A table row cannot be searched by meaning; a sentence can.

## ⭐ The anchors that landed — use these, they are his
| Idea | Anchor |
|---|---|
| Vector search | **NOT Ctrl+F.** Ctrl+F needs the exact word; this finds the meaning even when the word is different. |
| Why LEIE and not NPPES | **Two books.** NPPES = the phone book (everyone, no trouble). LEIE = who got caught and what they did. |
| Why not RAG on NPPES | If you already know which doctor, you don't search — **you look him up by his number.** |
| Retrieval | **The librarian.** *"I want the book about the boy who goes to a magic school"* — Ctrl+F needs the title, the librarian brings you three books. |
| Embedding | every sentence gets **an address on a map**; similar meanings land near each other |
| Closeness | how near two addresses are — **cosine similarity** |
| HNSW | **city → neighbourhood → street.** Big jumps first, smaller as it closes in. You never check every address. |

## The three things an interviewer WILL follow up with
Not the maths. **What an embedding is · how closeness is measured · how it searches fast.**
- **Cosine similarity, complete answer:** *"Two arrows. Same way — 1. Right angles — 0. Opposite — −1."*
- **HNSW** = **H**ierarchical **N**avigable **S**mall **W**orld. ⚠️ He typed "HSNW" — easy flip, correct it.
- **The cost of HNSW:** it is *approximate* — it can occasionally miss the true nearest. Traditional
  search checks every address and is exact but slow. **Naming that trade is the senior answer.**

## Terminology he got close on — do not re-blur
- **"Context search"** ❌ → it is **semantic search**. "Context" means something else: the retrieved
  sentences handed to the model are *the context*.
- **What is being searched:** the step-1 sentences, by their position on the meaning map.
