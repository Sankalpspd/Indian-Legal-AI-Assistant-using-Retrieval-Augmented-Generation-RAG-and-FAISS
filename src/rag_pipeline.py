# src/rag_pipeline.py

from src.embed_store import load_vectorstore
from src.hybrid_search import HybridRetriever
from src.definitions_retriever import detect_act, prioritize_definitions
from src.citation_generator import generate_citations
from transformers import pipeline

# Initialize offline summarizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_answer(text, max_len=150, min_len=40):
    """
    Summarize the raw concatenated text of retrieved documents.
    """
    if not text.strip():
        return "No relevant information found."
    summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]["summary_text"]


class LegalRAG:
    """
    Fully offline Legal AI using RAG + FAISS + BM25.
    Summarizes answers and provides citations.
    """
    def __init__(self):
        # Load FAISS vectorstore
        self.vectorstore = load_vectorstore()
        self.documents = list(self.vectorstore.docstore._dict.values())
        self.retriever = HybridRetriever(self.vectorstore, self.documents)
        self.acts_list = list(set([doc.metadata.get("act", "") for doc in self.documents]))

    def answer(self, query):
        """
        Returns:
        - summarized answer (string)
        - list of citations (Act - Section)
        """

        # Step 1: Detect relevant Act
        detected_act = detect_act(query, self.acts_list)
        filtered_docs = self.documents
        if detected_act:
            filtered_docs = [doc for doc in self.documents if doc.metadata.get("act", "").lower() == detected_act.lower()]

        # Step 2: Retrieve top documents (Hybrid retrieval)
        retrieved = self.retriever.search(query, docs=filtered_docs)

        # Step 3: Prioritize definition sections
        retrieved = prioritize_definitions(retrieved)

        # Step 4: Remove duplicates
        unique_docs = list({doc.page_content: doc for doc in retrieved}.values())

        # Step 5: Filter by section number if mentioned in query
        query_section = None
        words = query.lower().split()
        for w in words:
            if w.isdigit():
                query_section = w
                break

        if query_section:
            filtered_by_section = [doc for doc in unique_docs if query_section in doc.metadata.get("section", "")]
            top_docs = filtered_by_section[:5] if filtered_by_section else unique_docs[:5]
        else:
            top_docs = unique_docs[:5]

        # Step 6: Concatenate top documents
        raw_answer = "\n\n".join([doc.page_content for doc in top_docs])

        # Step 7: Summarize answer
        answer_text = summarize_answer(raw_answer)

        # Step 8: Generate citations
        citations = generate_citations(top_docs)

        return answer_text, citations
