import os
import re
import chromadb
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


def setup_rag(model_name, embed_model, text_file, chroma_path, chroma_col,
              prompt, chunk_size=1024, chunk_overlap=200, top_k=15, base_url=None):
    ollama_params = dict(
        model=model_name,
        request_timeout=1800.0,
        system_prompt=prompt,
        context_window=8000,
        temperature=0.0
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


def run_rag(query_engine, questions, architecture="naive", llm=None,
            xai_log_path=None, exec_num=1):
    """
    Returns list of (answer, hallucinations_count) tuples.
    hallucinations_count is -1 for naive architecture (not applicable).
    xai_log_path: path to save the XAI trace file (only used when architecture=xai)
    exec_num: current execution number (used for labeling in the trace file)
    """
    answers = []

    for i, q in enumerate(questions):

        # ─── FASE 1: User Query ───────────────────────────────────────────
        print(f"  [{i+1}/{len(questions)}] {q['question'][:60]}...")

        # ─── FASE 2: Document Retrieval ──────────────────────────────────
        # ─── FASE 3: Transparent Response Generation ─────────────────────
        # If architecture=xai, prompt instructs LLM to cite sources.
        response   = query_engine.query(q["question"])
        rag_answer = str(response).strip().replace('\n', ' ').replace(';', ',')
        hallucinations = -1  # not applicable for naive

        if architecture == "xai":

            # ─── FASE 4: Explainability Layer ────────────────────────────
            # Show which chunks ChromaDB actually retrieved and their scores.
            source_texts = [node.text for node in response.source_nodes]
            print(f"  Sources used:")
            for node in response.source_nodes:
                print(f"    - Score: {node.score:.3f} | {node.text[:100]}...")

            # ─── FASE 5: Feedback on Clarity ─────────────────────────────
            # Check citations and detect hallucinations.
            citation_found = "[Source:" in rag_answer or "[Fuente:" in rag_answer
            hallucinations = 0

            if citation_found:
                citations = re.findall(r'\[Source:(.*?)\]', rag_answer)
                for citation in citations:
                    citation_clean = citation.strip().lower()
                    found_in_chunks = any(
                        citation_clean[:40] in chunk.lower()
                        for chunk in source_texts
                    )
                    if not found_in_chunks:
                        print(f"  ⚠️  HALLUCINATION: '{citation.strip()[:60]}' not found in retrieved chunks")
                        hallucinations += 1

            # ─── FASE 6: Refine Explainability ───────────────────────────
            # If no citations or hallucinations detected, regenerate with
            # explicit chunks so the LLM must cite from them.
            if (not citation_found or hallucinations > 0) and llm:
                chunks_str    = "\n---\n".join(source_texts)
                refine_prompt = (
                    f"Question: {q['question']}\n\n"
                    f"Context fragments:\n{chunks_str}\n\n"
                    f"Answer the question using ONLY these fragments. "
                    f"Cite each one like [Source: exact fragment used]."
                )
                refined    = llm.complete(refine_prompt)
                rag_answer = str(refined).strip().replace('\n', ' ').replace(';', ',')

                # Re-check hallucinations after refinement
                hallucinations = 0
                citations = re.findall(r'\[Source:(.*?)\]', rag_answer)
                for citation in citations:
                    citation_clean = citation.strip().lower()
                    found_in_chunks = any(
                        citation_clean[:40] in chunk.lower()
                        for chunk in source_texts
                    )
                    if not found_in_chunks:
                        print(f"  ⚠️  HALLUCINATION after refinement: '{citation.strip()[:60]}'")
                        hallucinations += 1

            # ─── XAI TRACE LOG ───────────────────────────────────────────
            # Save full trace for this question: answer, chunks, citations.
            # First question of first exec clears the file; rest appends.
            if xai_log_path:
                mode = 'w' if (exec_num == 1 and i == 0) else 'a'
                with open(xai_log_path, mode, encoding='utf-8') as log:
                    log.write(f"\n{'='*80}\n")
                    log.write(f"EXEC {exec_num} | QUESTION {i+1}: {q['question']}\n")
                    log.write(f"{'='*80}\n")
                    log.write(f"\nANSWER:\n{rag_answer}\n")
                    log.write(f"\nHALLUCINATIONS DETECTED: {hallucinations}\n")
                    log.write(f"\nRETRIEVED CHUNKS ({len(source_texts)} total):\n")
                    for j, (node, text) in enumerate(zip(response.source_nodes, source_texts)):
                        log.write(f"\n  --- CHUNK {j+1} | Score: {node.score:.3f} ---\n")
                        log.write(f"  {text}\n")
                    log.write(f"\nCITATIONS FOUND IN ANSWER:\n")
                    citations_final = re.findall(r'\[Source:(.*?)\]', rag_answer)
                    if citations_final:
                        for citation in citations_final:
                            citation_clean = citation.strip().lower()
                            found = any(citation_clean[:40] in chunk.lower() for chunk in source_texts)
                            status = "✅ FOUND IN CHUNKS" if found else "❌ NOT FOUND — HALLUCINATION"
                            log.write(f"  [{status}] {citation.strip()[:80]}\n")
                    else:
                        log.write(f"  No citations found in answer\n")

        # ─── FASE 7: Final Output ─────────────────────────────────────────
        answers.append((rag_answer, hallucinations))

    return answers