{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cfeab295-88b1-4743-9872-7e8151496c09",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "from src.rag_pipeline import ask_legal_question\n",
    "\n",
    "\n",
    "st.set_page_config(\n",
    "    page_title=\"Indian Legal AI Assistant\",\n",
    "    page_icon=\"⚖️\",\n",
    "    layout=\"wide\"\n",
    ")\n",
    "\n",
    "st.title(\"⚖️ Indian Legal AI Assistant\")\n",
    "st.write(\"Ask questions about Indian law. The system retrieves relevant legal sections and generates answers with citations.\")\n",
    "\n",
    "\n",
    "query = st.text_input(\"Enter your legal question:\")\n",
    "\n",
    "\n",
    "if st.button(\"Ask\"):\n",
    "\n",
    "    if query.strip() == \"\":\n",
    "        st.warning(\"Please enter a question.\")\n",
    "    else:\n",
    "\n",
    "        with st.spinner(\"Searching legal documents...\"):\n",
    "\n",
    "            answer, citations = ask_legal_question(query)\n",
    "\n",
    "        st.subheader(\"Answer\")\n",
    "        st.write(answer)\n",
    "\n",
    "        st.subheader(\"Citations\")\n",
    "\n",
    "        if citations:\n",
    "            for c in citations:\n",
    "                st.write(f\"- {c}\")\n",
    "        else:\n",
    "            st.write(\"No citations found.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
