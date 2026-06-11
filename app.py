"""Web UI — AI Document Assistant (Streamlit).

Run:  streamlit run app.py

Features (grow with the lessons):
- upload PDF/Word/Excel/PPT -> ingest -> index into Qdrant
- ask questions -> streamed RAG answer with citations

All UI strings come from config/ui.yaml (no hardcoded text). Works once the
provider backend (local Ollama by default) and Qdrant are running.

Streamlit re-runs this whole script on every interaction, so the heavy
embedding model is cached with @st.cache_resource and loaded only once.
"""
import tempfile
from pathlib import Path

import streamlit as st

from docassistant.config import get_ui_text
from docassistant.ingest import SUPPORTED, load_and_split
from docassistant.rag import answer_stream
from docassistant.store import add_documents, get_retriever


@st.cache_resource(show_spinner=False)
def cached_retriever():
    """Build the retriever once (loads the embedding model) and reuse it.

    Qdrant is queried live each call, so newly indexed documents are still
    visible — only the embedding model load is cached.
    """
    return get_retriever()


st.set_page_config(
    page_title=get_ui_text("page_title"),
    page_icon=get_ui_text("page_icon"),
)
st.title(get_ui_text("header"))

with st.sidebar:
    st.header(get_ui_text("sidebar_header"))
    uploaded = st.file_uploader(
        get_ui_text("uploader_label"),
        type=[e.lstrip(".") for e in SUPPORTED],
        accept_multiple_files=True,
    )
    if st.button(get_ui_text("index_button"), disabled=not uploaded):
        total = 0
        for uf in uploaded:
            suffix = Path(uf.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.getbuffer())
                tmp_path = Path(tmp.name)
            try:
                chunks = load_and_split(tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)  # never leave temp files behind
            # keep the original filename as the source (tmp has a random name)
            for c in chunks:
                c.metadata["source"] = uf.name
            total += add_documents(chunks)
        st.success(get_ui_text("index_success").format(chunks=total, files=len(uploaded)))

st.subheader(get_ui_text("ask_subheader"))
question = st.text_input(get_ui_text("question_label"))
if st.button(get_ui_text("answer_button"), disabled=not question):
    with st.spinner(get_ui_text("answer_spinner")):
        tokens, sources = answer_stream(question, retriever=cached_retriever())
    st.write_stream(tokens)  # answer appears token-by-token
    with st.expander(get_ui_text("sources_expander")):
        for i, d in enumerate(sources, start=1):
            src = d.metadata.get("source", "?")
            page = d.metadata.get("page")
            loc = f"{src}, p.{page}" if page is not None else src
            st.markdown(f"**[{i}] {loc}**")
            st.caption(d.page_content[:400] + "...")
