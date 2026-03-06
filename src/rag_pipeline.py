from src.embed_store import load_vectorstore
from src.hybrid_search import HybridRetriever
from src.definitions_retriever import prioritize_definitions, detect_act
from src.citation_generator import generate_citations

class LegalRAG:
    def __init__(self):
        # Load FAISS vector store
        self.vectorstore = load_vectorstore()
        self.documents = list(self.vectorstore.docstore._dict.values())
        # Hybrid retriever (BM25 + embeddings)
        self.retriever = HybridRetriever(self.vectorstore, self.documents)
        # List of all acts for section-aware filtering
        self.acts_list = list(set([doc.metadata.get("act", "") for doc in self.documents]))

    def answer(self, query):
        """
        Return retrieved sections and citations for a legal question.
        No LLM is used. Pure retrieval + citation.
        """

        # Step 1: Detect relevant act
        detected_act = detect_act(query, self.acts_list)
        filtered_docs = self.documents
        if detected_act:
            filtered_docs = [doc for doc in self.documents if doc.metadata.get("act", "").lower() == detected_act.lower()]

        # Step 2: Retrieve documents using hybrid retriever
        retrieved = self.retriever.search(query, docs=filtered_docs)

        # Step 3: Prioritize definitions sections
        retrieved = prioritize_definitions(retrieved)

        # Step 4: Remove duplicate sections
        unique_docs = list({doc.page_content: doc for doc in retrieved}.values())

        # Step 5: Build "answer" as concatenation of top sections
        top_docs = unique_docs[:5]  # top 5 results
        answer_text = "\n\n".join([doc.page_content for doc in top_docs])

        # Step 6: Generate citations
        citations = generate_citations(top_docs)

        return answer_text, citations
