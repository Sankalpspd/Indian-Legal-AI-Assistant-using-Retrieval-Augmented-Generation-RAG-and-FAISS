import streamlit as st
from src.rag_pipeline import LegalRAG

# Page config
st.set_page_config(
    page_title="Indian Legal AI Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Indian Legal AI Assistant")
st.write("Ask questions about Indian laws and acts.")

# Load RAG system
@st.cache_resource
def load_rag():
    return LegalRAG()

rag = load_rag()

# User input
query = st.text_input("Enter your legal question:")

if st.button("Ask"):

    if query.strip() == "":
        st.warning("Please enter a question.")
    else:

        with st.spinner("Searching legal documents..."):
            answer, citations = rag.answer(query)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")

        if citations:
            for c in citations:
                st.write("•", c)
        else:
            st.write("No citations found.")
