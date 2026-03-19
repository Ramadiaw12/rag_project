import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

load_dotenv(override=True)

# ─────────────────────────────────────────────
#  Page config (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG · DocMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Night-mode CSS  (forced, no toggle)
# ─────────────────────────────────────────────
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Tokens ───────────────────────────────── */
:root {
    --accent:      #818CF8;
    --accent-h:    #A5B4FC;
    --accent-dim:  #1E1B4B;
    --bg-base:     #0D0D14;
    --bg-surface:  #13131E;
    --bg-raised:   #1A1A2E;
    --bg-input:    #1E1E30;
    --border:      rgba(255,255,255,.08);
    --border-mid:  rgba(255,255,255,.14);
    --text:        #E2E2EE;
    --muted:       #6B7280;
    --success:     #10B981;
    --warning:     #F59E0B;
    --danger:      #EF4444;
    --r:           12px;
    --rs:          8px;
    --glow:        0 0 28px rgba(129,140,248,.18);
}

/* ── Reset / base ─────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}
h1,h2,h3,h4 { font-family:'Syne',sans-serif !important; color:var(--text) !important; }
#MainMenu,footer,header { visibility:hidden; }
.block-container { padding-top:1.5rem !important; }

/* ── Force dark everywhere ────────────────── */
.stApp,
[data-theme="light"] .stApp,
[data-theme="dark"]  .stApp {
    background: var(--bg-base) !important;
}
[data-testid="stAppViewContainer"] > .main { background: var(--bg-base) !important; }
section[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .block-container { padding-top:2rem; }
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
.element-container { background: transparent !important; }

/* ── Expander ─────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p { color: var(--text) !important; }

/* ── Inputs ───────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea  > div > div > textarea,
[data-baseweb="input"] input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-mid) !important;
    color: var(--text) !important;
    border-radius: var(--rs) !important;
    font-family: 'DM Sans',sans-serif !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea  > div > div > textarea::placeholder { color:var(--muted) !important; }
[data-baseweb="form-control-label"] { color:var(--text) !important; }

/* ── File uploader ────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--bg-raised) !important;
    border: 1.5px dashed var(--border-mid) !important;
    border-radius: var(--r) !important;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p { color: var(--muted) !important; }

/* ── Buttons ──────────────────────────────── */
.stButton > button {
    border-radius: var(--rs) !important;
    font-family: 'DM Sans',sans-serif !important;
    font-weight: 500 !important;
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-mid) !important;
    color: var(--text) !important;
    transition: border-color .2s, color .2s;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#4338CA,#6D28D9) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 4px 16px rgba(109,40,217,.45) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg,#4F46E5,#7C3AED) !important;
    color: #fff !important;
    box-shadow: 0 6px 24px rgba(109,40,217,.55) !important;
}

/* ── Slider ───────────────────────────────── */
[data-testid="stSlider"] label { color:var(--text) !important; }

/* ── Scrollbar ────────────────────────────── */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: #2A2A3E; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Cards ────────────────────────────────── */
.rag-card {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: box-shadow .25s, border-color .25s;
}
.rag-card:hover { border-color:var(--border-mid); box-shadow:var(--glow); }
.rag-card p, .rag-card span, .rag-card li { color:var(--text) !important; }

/* ── Metric cards ─────────────────────────── */
.metric-card {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1rem 1.25rem;
    text-align: center;
    transition: box-shadow .25s, border-color .25s;
}
.metric-card:hover { border-color:var(--accent); box-shadow:var(--glow); }
.metric-value { font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:700; color:var(--accent); }
.metric-label { font-size:.78rem; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; margin-top:.2rem; }

/* ── Hero ─────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg,#1E1B5E 0%,#2D1B69 100%);
    border: 1px solid rgba(129,140,248,.22);
    border-radius: var(--r);
    padding: 1.75rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--glow);
}
.hero-banner h1 { color:#fff !important; font-size:1.9rem; margin:0 0 .35rem; }
.hero-banner p  { color:rgba(255,255,255,.72) !important; margin:0; font-size:.95rem; }

/* ── Chat bubbles ─────────────────────────── */
.chat-bubble-user {
    background: linear-gradient(135deg,#3730A3,#5B21B6);
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: .75rem 1.1rem;
    margin: .5rem 0 .5rem auto;
    max-width: 80%;
    width: fit-content;
    font-size: .93rem;
    box-shadow: 0 4px 16px rgba(91,33,182,.4);
}
.chat-bubble-ai {
    background: var(--bg-raised);
    border: 1px solid var(--border-mid);
    color: var(--text);
    border-radius: 18px 18px 18px 4px;
    padding: .75rem 1.1rem;
    margin: .5rem auto .5rem 0;
    max-width: 90%;
    width: fit-content;
    font-size: .93rem;
}
.chat-sender           { font-size:.72rem; font-weight:600; margin-bottom:.3rem; letter-spacing:.05em; }
.chat-sender.user      { color:rgba(255,255,255,.6); }
.chat-sender.ai        { color:var(--accent); }

/* ── Source pills ─────────────────────────── */
.source-pill {
    display: inline-block;
    background: var(--accent-dim);
    color: var(--accent);
    border-radius: 20px;
    padding: .2rem .75rem;
    font-size: .78rem;
    font-weight: 500;
    margin: .25rem .25rem 0 0;
}

/* ── Doc badges ───────────────────────────── */
.doc-badge {
    display: flex;
    align-items: center;
    gap: .6rem;
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: var(--rs);
    padding: .55rem .9rem;
    margin: .4rem 0;
    font-size: .85rem;
    color: var(--text);
}
.doc-icon { font-size:1.1rem; }

/* ── Divider ──────────────────────────────── */
.rag-divider { border:none; border-top:1px solid var(--border); margin:1.25rem 0; }

/* ── Status dots ──────────────────────────── */
.status-dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:6px; }
.status-dot.ok    { background:var(--success); box-shadow:0 0 8px rgba(16,185,129,.55); }
.status-dot.warn  { background:var(--warning); }
.status-dot.error { background:var(--danger); }

/* ── Form container ───────────────────────── */
[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ── Alerts ───────────────────────────────── */
.stAlert { border-radius: var(--r) !important; }
[data-testid="stMarkdownContainer"] p { color:var(--text) !important; }
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────
if "retriever"    not in st.session_state: st.session_state.retriever    = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "docs_meta"    not in st.session_state: st.session_state.docs_meta    = []
if "total_chunks" not in st.session_state: st.session_state.total_chunks = 0

# ─────────────────────────────────────────────
#  LLM & prompt
# ─────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a precise, helpful assistant that answers questions strictly based on the provided context.
If the context does not contain sufficient information to answer, say so clearly.

<context>
{context}
</context>

<question>
{input}
</question>

Provide a clear, well-structured answer. If relevant, mention which part of the document supports your answer."""

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0)

llm = get_llm()

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def process_pdfs(pdf_docs, chunk_size=512, chunk_overlap=64, top_k=5):
    full_text = ""
    meta = []
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        pages = len(reader.pages)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        full_text += text
        meta.append({"name": pdf.name, "pages": pages})

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(full_text)
    for m in meta:
        m["chunks"] = max(1, len(chunks) // max(len(meta), 1))

    embedding_model = OpenAIEmbeddings()
    vector_store = Chroma.from_texts(chunks, embedding_model, collection_name="rag_collection")
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    return retriever, chunks, meta


def answer_question(question: str):
    context_docs = st.session_state.retriever.invoke(question)
    context_text = "\n\n".join(d.page_content for d in context_docs)
    sources       = [d.page_content[:120] + "…" for d in context_docs]
    prompt = PROMPT_TEMPLATE.format(context=context_text, input=question)
    resp   = llm.invoke(prompt)
    return resp.content, sources


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """<div style='text-align:center;padding:0 0 1.2rem;'>
            <span style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;
                         background:linear-gradient(135deg,#818CF8,#A78BFA);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
                🧠 DocMind
            </span>
            <p style='color:#6B7280;font-size:.78rem;margin:.3rem 0 0;'>RAG · Powered by GPT-4o</p>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("#### 📂 Load Documents")
    pdf_docs = st.file_uploader(
        "Drop your PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if pdf_docs:
        st.markdown("**Files selected:**")
        for f in pdf_docs:
            size_kb = round(f.size / 1024, 1)
            st.markdown(
                f'<div class="doc-badge"><span class="doc-icon">📄</span>'
                f'<span>{f.name}</span>'
                f'<span style="margin-left:auto;color:#6B7280;font-size:.75rem;">{size_kb} KB</span></div>',
                unsafe_allow_html=True,
            )

    # Settings before process button
    with st.expander("⚙️ Settings"):
        top_k        = st.slider("Chunks retrieved (k)", 1, 10, 5)
        chunk_size   = st.slider("Chunk size (tokens)",  256, 1024, 512, step=64)
        chunk_overlap= st.slider("Chunk overlap",        0,   128,  64,  step=16)
        st.caption("Applied on next 'Process & Index'.")

    process_btn = st.button(
        "⚡ Process & Index",
        type="primary",
        use_container_width=True,
        disabled=not pdf_docs,
    )

    if process_btn and pdf_docs:
        with st.spinner("Extracting · Splitting · Embedding…"):
            try:
                retriever, chunks, meta = process_pdfs(
                    pdf_docs, chunk_size, chunk_overlap, top_k
                )
                st.session_state.retriever    = retriever
                st.session_state.docs_meta    = meta
                st.session_state.total_chunks = len(chunks)
                st.session_state.chat_history = []
                st.success(f"✅ Indexed {len(chunks)} chunks from {len(meta)} document(s)!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.markdown('<hr class="rag-divider">', unsafe_allow_html=True)

    if st.session_state.retriever:
        st.markdown(
            '<span class="status-dot ok"></span>**Index ready** — ask your questions!',
            unsafe_allow_html=True,
        )
        for doc in st.session_state.docs_meta:
            st.markdown(
                f'<div class="doc-badge">'
                f'<span class="doc-icon">✅</span>'
                f'<span style="font-size:.83rem;">{doc["name"]}</span>'
                f'<span style="margin-left:auto;color:#6B7280;font-size:.75rem;">{doc["pages"]} p.</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<span class="status-dot warn"></span>No index loaded yet.',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="rag-divider">', unsafe_allow_html=True)

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown(
    """<div class="hero-banner">
        <h1>DocMind — Retrieval Augmented Generation</h1>
        <p>Upload PDFs · Index in seconds · Ask anything · Get answers grounded in your documents.</p>
    </div>""",
    unsafe_allow_html=True,
)

# ── Metrics ──
docs_count      = len(st.session_state.docs_meta)
total_pages     = sum(d.get("pages", 0) for d in st.session_state.docs_meta)
total_questions = sum(1 for m in st.session_state.chat_history if m["role"] == "user")

c1, c2, c3, c4 = st.columns(4)
for col, val, label in zip(
    [c1, c2, c3, c4],
    [docs_count, total_pages, st.session_state.total_chunks, total_questions],
    ["Documents indexed", "Total pages", "Text chunks", "Questions asked"],
):
    col.markdown(
        f'<div class="metric-card"><div class="metric-value">{val}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Two-column layout ──
chat_col, ctx_col = st.columns([2, 1], gap="large")

with chat_col:
    st.markdown("#### 💬 Conversation")
    chat_container = st.container(height=480)
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                '<div style="text-align:center;padding:3rem 0;color:#4B5563;">'
                '<p style="font-size:2rem;margin-bottom:.5rem;">💬</p>'
                '<p style="color:#4B5563;">Ask a question about your documents.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-bubble-user">'
                    f'<div class="chat-sender user">YOU</div>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                sources_html = "".join(
                    f'<span class="source-pill">…{s[-60:]}</span>'
                    for s in msg.get("sources", [])[:3]
                )
                st.markdown(
                    f'<div class="chat-bubble-ai">'
                    f'<div class="chat-sender ai">DOCMIND</div>{msg["content"]}'
                    f'<div style="margin-top:.6rem;">{sources_html}</div></div>',
                    unsafe_allow_html=True,
                )

    with st.form("question_form", clear_on_submit=True):
        user_q = st.text_input(
            "Your question",
            placeholder="e.g.  What are the main findings of the report?",
            label_visibility="collapsed",
        )
        col_send, col_hint = st.columns([1, 3])
        with col_send:
            submitted = st.form_submit_button("Send ➤", type="primary", use_container_width=True)
        with col_hint:
            if not st.session_state.retriever:
                st.caption("⚠️ Index your documents first.")

    if submitted and user_q:
        if not st.session_state.retriever:
            st.warning("Please upload and index your documents first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.spinner("Thinking…"):
                try:
                    answer, sources = answer_question(user_q)
                    st.session_state.chat_history.append(
                        {"role": "ai", "content": answer, "sources": sources}
                    )
                except Exception as e:
                    st.session_state.chat_history.append(
                        {"role": "ai", "content": f"⚠️ Error: {e}", "sources": []}
                    )
            st.rerun()

with ctx_col:
    st.markdown("#### 🔍 Source Context")
    last_ai = next(
        (m for m in reversed(st.session_state.chat_history) if m["role"] == "ai"), None
    )
    if last_ai and last_ai.get("sources"):
        for i, src in enumerate(last_ai["sources"], 1):
            with st.expander(f"Chunk #{i}", expanded=(i == 1)):
                st.markdown(
                    f'<div class="rag-card" style="margin-bottom:0;">'
                    f'<p style="font-size:.85rem;line-height:1.65;">{src}</p></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            '<div style="text-align:center;padding:2.5rem 0;color:#4B5563;">'
            '<p style="font-size:1.5rem;">🔎</p>'
            '<p style="font-size:.9rem;color:#4B5563;">Retrieved context appears here after your first question.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="rag-divider">', unsafe_allow_html=True)
    st.markdown("#### ℹ️ How it works")
    st.markdown
    """<div class="rag-card" style="font-size:.85rem;line-height:1.75;">
<strong style="color:#818CF8;">1. 📄 Upload</strong> — Load one or more PDF documents.<br><br>
<strong style="color:#818CF8;">2. ⚡ Index</strong> — Text is extracted, split into overlapping chunks and stored as vector embeddings (OpenAI + Chroma).<br><br>
<strong style="color:#818CF8;">3. 💬 Ask</strong> — Your question is embedded; the top-<em>k</em> most similar chunks are retrieved.<br><br>
<strong style="color:#818CF8;">4. 🧠 Generate</strong> — GPT-4o uses <em>only</em> those chunks to produce a grounded answer.
</div>"""