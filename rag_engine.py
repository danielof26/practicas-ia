import os
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
        request_timeout=1200.0,
        system_prompt=prompt,
        context_window=8000,
        temperature=0.1
    )

    if base_url:
        ollama_params["base_url"] = base_url

    llm = Ollama(**ollama_params)

    Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    Settings.embed_model = OllamaEmbedding(model_name=embed_model)
    Settings.llm = llm

    db = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = db.get_or_create_collection(chroma_col)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if chroma_collection.count() == 0:
        print(f"Indexing document: {text_file}")
        documents = SimpleDirectoryReader(input_files=[text_file]).load_data()
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context
        )

        print("Document indexed correctly.")
    else:
        index = VectorStoreIndex.from_vector_store(
            vector_store, 
            storage_context=storage_context
        )
    
    query_engine = index.as_query_engine(similarity_top_k=top_k)

    return query_engine



def run_rag(query_engine, questions):

    answers = []
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['question'][:60]}...")
        response = query_engine.query(q["question"])
        rag_answer = str(response).strip().replace('\n', ' ').replace(';', ',')
        answers.append(rag_answer)
    
    return answers
