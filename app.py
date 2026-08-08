"""
Streamlit UI for the local RAG assistant (Week 4 "Option B").

Run:
    streamlit run app.py

Everything here runs on-device via Foundry Local — no data leaves your
computer, and after the first run (which downloads the models once) the
app works fully offline.
"""

import streamlit as st

import config
import db
import ingest
import llm
import retrieval

st.set_page_config(page_title="Local RAG Assistant", page_icon="🤖")


@st.cache_resource(show_spinner=False)
def _init_models():
    """Load Foundry Local models once per app session."""
    llm.initialize()
    return True


def _ensure_db_ready():
    db.init_db()
    return db.count_chunks()


st.title("🤖 Local RAG Assistant")
st.caption("Runs 100% on your device with Microsoft Foundry Local — no internet required after setup.")

with st.sidebar:
    st.header("Knowledge base")
    n_chunks = _ensure_db_ready()
    st.write(f"Indexed chunks: **{n_chunks}**")
    st.write(f"Documents folder: `{config.DOCUMENTS_DIR}`")
    st.caption("Add .txt / .md files there, then click below.")

    if st.button("🔄 (Re)build knowledge base", use_container_width=True):
        progress_bar = st.progress(0.0, text="Starting...")

        def _cb(stage, pct):
            progress_bar.progress(min(pct / 100, 1.0), text=f"{stage} ({pct:.0f}%)")

        with st.spinner("Ingesting documents (first run downloads models, can take a few minutes)..."):
            try:
                count = ingest.run_ingestion(progress_callback=_cb)
                st.success(f"Indexed {count} chunks.")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()
    st.caption(f"Embedding model: `{config.EMBEDDING_MODEL_ALIAS}`")
    st.caption(f"Chat model: `{config.CHAT_MODEL_ALIAS}`")

if n_chunks == 0:
    st.info("No documents indexed yet. Add files to the `documents/` folder and click "
             "**Rebuild knowledge base** in the sidebar to get started.")
    st.stop()

with st.spinner("Loading local models (first run downloads them, this may take a while)..."):
    _init_models()

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn.get("sources"):
            with st.expander("Sources used"):
                for s in turn["sources"]:
                    st.markdown(f"**{s['source']}** (score {s['score']:.2f})\n\n> {s['content']}")

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    query_embedding = llm.embed_query(question)
    top_chunks = retrieval.get_top_chunks(query_embedding)
    best_score = top_chunks[0]["score"] if top_chunks else 0.0

    with st.chat_message("assistant"):
        if best_score < config.MIN_RELEVANCE_SCORE:
            # Nothing relevant enough was found — don't even ask the chat
            # model, since small local models sometimes answer from general
            # knowledge instead of admitting they don't know.
            full_answer = (
                "I don't have information about that in the indexed documents."
            )
            st.write(full_answer)
            top_chunks = []
        else:
            context = "\n\n".join(f"[{c['source']}] {c['content']}" for c in top_chunks)
            system_prompt = config.SYSTEM_PROMPT_TEMPLATE.format(context=context)

            placeholder = st.empty()
            full_answer = ""
            for token in llm.answer_stream(system_prompt, question):
                full_answer += token
                placeholder.write(full_answer)

            with st.expander("Sources used"):
                for c in top_chunks:
                    st.markdown(f"**{c['source']}** (score {c['score']:.2f})\n\n> {c['content']}")

    st.session_state.history.append(
        {"role": "assistant", "content": full_answer, "sources": top_chunks}
    )
