from src.embed_store import load_vectorstore
from src.hybrid_search import HybridRetriever
from src.query_rewriter import rewrite_query
from src.citation_generator import generate_citations
from src.definitions_retriever import prioritize_definitions
from transformers import pipeline


class LegalRAG:

    def __init__(self):

        # Load FAISS vector database
        self.vectorstore = load_vectorstore()

        # Get all stored documents
        self.documents = list(self.vectorstore.docstore._dict.values())

        # Hybrid retriever (BM25 + FAISS)
        self.retriever = HybridRetriever(self.vectorstore, self.documents)

        # Local LLM
        self.generator = pipeline(
            "text-generation",
            model="google/flan-t5-base",   # lightweight model
            max_new_tokens=300
        )

        # List of Acts for filtering
        self.acts_list = list(set([doc.metadata.get("act", "") for doc in self.documents]))


    def answer(self, query):

        # STEP 1 — Query rewriting
        queries = rewrite_query(query)

        # STEP 2 — Retrieval
        retrieved_docs = []

        for q in queries:
            results = self.retriever.search(q)
            retrieved_docs.extend(results)

        # Remove duplicate chunks
        unique_docs = {doc.page_content: doc for doc in retrieved_docs}
        retrieved_docs = list(unique_docs.values())

        # STEP 3 — Prioritize legal definitions
        retrieved_docs = prioritize_definitions(retrieved_docs)

        # Take top documents
        top_docs = retrieved_docs[:5]

        # STEP 4 — Build context
        context = "\n\n".join([doc.page_content for doc in top_docs])

        prompt = f"""
You are an AI legal assistant.

Use ONLY the legal context below to answer the question.

If the answer is not in the context, say you cannot find it.

Context:
{context}

Question:
{query}

Answer clearly and include Act and Section references when possible.
"""

        # STEP 5 — Generate answer
        response = self.generator(prompt)

        answer = response[0]["generated_text"]

        # STEP 6 — Generate citations
        citations = generate_citations(top_docs)

        return answer, citations
