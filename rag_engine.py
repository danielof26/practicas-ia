import os
import re
import chromadb
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

_TRACE_SEP = "=" * 80


def setup_rag(model_name, embed_model, text_file, chroma_path, chroma_col,
              prompt, chunk_size=1024, chunk_overlap=200, top_k=15, base_url=None,
              context_window=8000):
    """Initialises the Ollama LLM, ChromaDB vector store and LlamaIndex query engine."""
    ollama_params = dict(
        model=model_name,
        request_timeout=1800.0,
        system_prompt=prompt,
        context_window=context_window,
        temperature=0.1
    )
    if base_url:
        ollama_params["base_url"] = base_url
    llm = Ollama(**ollama_params)

    Settings.node_parser  = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    Settings.embed_model  = OllamaEmbedding(model_name=embed_model)
    Settings.llm          = llm

    db                = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = db.get_or_create_collection(chroma_col)
    vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context   = StorageContext.from_defaults(vector_store=vector_store)

    if chroma_collection.count() == 0:
        print(f"Indexing document: {text_file}")
        documents = SimpleDirectoryReader(input_files=[text_file]).load_data()
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        print("Document indexed correctly.")
    else:
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    query_engine = index.as_query_engine(similarity_top_k=top_k)
    return query_engine, llm


def _citation_in_chunks(citation: str, source_texts: list, threshold: float = 0.6) -> bool:
    '''
        Returns True if ≥threshold fraction of citation words appear in any chunk.
        Punctuation is stripped before comparison to avoid false mismatches (e.g. commas vs semicolons).
    '''
    clean = lambda text: re.sub(r'[^\w\s]', '', text.lower())
    citation_words = set(clean(citation).split())
    if not citation_words:
        return False
    for chunk in source_texts:
        chunk_words = set(clean(chunk).split())
        overlap = len(citation_words & chunk_words) / len(citation_words)
        if overlap >= threshold:
            return True
    return False



def _check_citations(rag_answer: str, source_texts: list):
    '''
        Centralised citation check used by the feedback loop.
        Returns (all_citations, bad_citations, hallucinations_count).
        hallucinations=-2 when model produced no [Source:] tags at all.
    '''
    citations = re.findall(r'\[Source:(.*?)\]', rag_answer)
    real = [c for c in citations if "verbatim text copied from context" not in c]
    if not real:
        return [], [], -2
    bad = [c for c in real if not _citation_in_chunks(c, source_texts)]
    return real, bad, len(bad)


def _report_bad_citations(bad_citations: list, attempt: int):
    """Prints terminal warnings for unverified citations, ignoring placeholder entries."""
    for bc in bad_citations:
        print(f"  ⚠️  HALLUCINATION (attempt {attempt}): '{bc.strip()[:60]}'")


def _refine_answer(llm, question: str, source_texts: list) -> str:
    '''Re-prompts the LLM with raw chunks when citations fail verification (Phase 6).'''
    chunks_str    = "\n---\n".join(source_texts)
    refine_prompt = (
        f"Question: {question}\n\n"
        f"Context fragments:\n{chunks_str}\n\n"
        f"Answer the question using ONLY these fragments. "
        f"Cite each one like [Source: <verbatim text copied from context>]."
    )
    return str(llm.complete(refine_prompt)).strip().replace('\n', ' ').replace(';', ',')



def _xai_feedback_loop(rag_answer: str, source_texts: list, llm, question: str,
                        max_refine: int = 2):
    '''
        Implements the Phase 5→6 check→refine loop.
        Returns (final_answer, citations, hallucinations_count, process_log).
        process_log records every intermediate attempt for the XAI trace.
    '''
    citations    = []
    hallucinations = -1
    process_log  = []

    for attempt in range(max_refine + 1):
        # ─── FASE 5: Feedback on Clarity ─────────────────────────────────
        citations, bad_citations, hallucinations = _check_citations(rag_answer, source_texts)

        _report_bad_citations(bad_citations, attempt)

        needs_refine = hallucinations != 0  # -2 = no citations; >0 = bad citations

        if not needs_refine:
            process_log.append({
                "attempt":       attempt,
                "hallucinations": hallucinations,
                "bad_citations": bad_citations,
                "action":        "accepted"
            })
            break

        if attempt == max_refine or not llm:
            action = "exhausted" if attempt == max_refine else "no_llm"
            process_log.append({
                "attempt":        attempt,
                "hallucinations": hallucinations,
                "bad_citations":  bad_citations,
                "action":         action
            })
            if hallucinations == -2:
                print(f"  ⚠️  WARNING: No citations after {max_refine} refinement attempt(s) — answer not verifiable")
            break

        # ─── FASE 6: Refine Explainability ───────────────────────────────
        process_log.append({
            "attempt":        attempt,
            "hallucinations": hallucinations,
            "bad_citations":  bad_citations,
            "action":         "refined"
        })
        rag_answer = _refine_answer(llm, question, source_texts)
        print(f"  [Refinement {attempt + 1}/{max_refine}] regenerated answer...")

    return rag_answer, citations, hallucinations, process_log


_ACTION_LABELS = {
    "accepted":  "→ answer accepted ✅",
    "refined":   "→ REFINEMENT triggered 🔄",
    "exhausted": "→ max refinements reached ⛔",
    "no_llm":    "→ no LLM available ⛔",
}


def _format_process_entry(entry: dict) -> str:
    """Formats a single process log entry as a human-readable string."""
    attempt, h, action = entry["attempt"], entry["hallucinations"], entry["action"]
    if h == -2:
        status = f"  Attempt {attempt}: no citations found"
    elif h == 0:
        status = f"  Attempt {attempt}: 0 unverified citations"
    else:
        status = f"  Attempt {attempt}: {h} unverified citation(s) detected"
    return f"{status} {_ACTION_LABELS.get(action, '')}"


def _write_process_log(log, process_log: list):
    """Writes the Phase 5-6 process log section to the XAI trace file."""
    log.write("\nPROCESS LOG (Phases 5 & 6):\n")
    for entry in process_log:
        log.write(f"{_format_process_entry(entry)}\n")
        for bc in entry["bad_citations"]:
            log.write(f"    ⚠️  HALLUCINATION: [Source:{bc.strip()}]\n")

    intermediate_hallucinations = sum(
        e["hallucinations"] for e in process_log
        if e["hallucinations"] > 0
    )
    if intermediate_hallucinations > 0:
        fixed = process_log[-1]["hallucinations"] == 0
        status = "fixed by refinement ✅" if fixed else "NOT fixed ❌"
        log.write(f"\n  ⚠️  {intermediate_hallucinations} intermediate hallucination(s) detected — {status}\n")


def _write_citations_section(log, citations: list, source_texts: list):
    """Writes the citations verification summary to the XAI trace file."""
    log.write("\nCITATIONS FOUND IN ANSWER:\n")
    if citations:
        for citation in citations:
            found  = _citation_in_chunks(citation, source_texts)
            status = "✅ FOUND IN CHUNKS" if found else "❌ NOT FOUND — HALLUCINATION"
            log.write(f"  [{status}] {citation.strip()}\n")
    else:
        log.write("  No citations found — COMPLIANCE FAILURE\n")


def _write_xai_trace(log_path: str, exec_num: int, q_idx: int, question: str,
                     rag_answer: str, hallucinations: int,
                     response, source_texts: list, citations: list,
                     process_log: list):
    """Appends a full XAI trace entry for one question to the log file."""
    mode = 'w' if (exec_num == 1 and q_idx == 0) else 'a'
    with open(log_path, mode, encoding='utf-8') as log:
        log.write(f"\n{_TRACE_SEP}\n")
        log.write(f"EXEC {exec_num} | QUESTION {q_idx + 1}: {question}\n")
        log.write(f"{_TRACE_SEP}\n")

        _write_process_log(log, process_log)

        log.write(f"\nFINAL ANSWER:\n{rag_answer}\n")
        if hallucinations == -2:
            log.write("\nHALLUCINATIONS DETECTED (final): INDETERMINATE (model did not cite)\n")
        else:
            log.write(f"\nHALLUCINATIONS DETECTED (final): {hallucinations}\n")

        log.write(f"\nRETRIEVED CHUNKS ({len(source_texts)} total):\n")
        for j, (node, text) in enumerate(zip(response.source_nodes, source_texts)):
            log.write(f"\n  --- CHUNK {j + 1} | Score: {node.score:.3f} ---\n")
            log.write(f"  {text}\n")

        _write_citations_section(log, citations, source_texts)


def run_rag(query_engine, questions, architecture="naive", llm=None,
            xai_log_path=None, exec_num=1):
    """
    Returns list of (answer, hallucinations_count) tuples.

    hallucinations_count values:
      -1 → naive architecture (not applicable)
      -2 → XAI: model never produced citations even after all refinement attempts
       0 → XAI: all citations verified against retrieved chunks
      >0 → XAI: number of unverified citations (potential hallucinations)

    xai_log_path: path to save the XAI trace file (only used when architecture=xai)
    exec_num: current execution number (used for labeling in the trace file)
    """
    answers = []

    for i, q in enumerate(questions):

        # ─── FASE 1: User Query ───────────────────────────────────────────
        print(f"  [{i+1}/{len(questions)}] {q['question'][:60]}...")

        # ─── FASE 2: Document Retrieval ──────────────────────────────────
        # ─── FASE 3: Transparent Response Generation ─────────────────────
        # Inject citation instruction at query level to keep the system prompt theme-agnostic.
        question_text = q["question"]
        if architecture == "xai":
            question_text += (
                "\n\nIMPORTANT: For each statement, cite the exact source fragment "
                "like this: [Source: <verbatim text copied from context>]"
            )

        response       = query_engine.query(question_text)
        rag_answer     = str(response).strip().replace('\n', ' ').replace(';', ',')
        hallucinations = -1
        citations      = []

        if architecture == "xai":

            # ─── FASE 4: Explainability Layer ────────────────────────────
            source_texts = [node.text for node in response.source_nodes]
            print("  Sources used:")
            for node in response.source_nodes:
                print(f"    - Score: {node.score:.3f} | {node.text[:100]}...")

            # ─── FASE 5 & 6: Feedback on Clarity + Refine Explainability ─
            rag_answer, citations, hallucinations, process_log = _xai_feedback_loop(
                rag_answer, source_texts, llm, q["question"]
            )

            # ─── XAI TRACE LOG ───────────────────────────────────────────
            if xai_log_path:
                _write_xai_trace(
                    xai_log_path, exec_num, i, q["question"],
                    rag_answer, hallucinations, response, source_texts, citations,
                    process_log
                )

        # ─── FASE 7: Final Output ─────────────────────────────────────────
        answers.append((rag_answer, hallucinations))

    return answers
