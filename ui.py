import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide"
)

# ── Sidebar navigation ──────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/open-book.png", width=60)
st.sidebar.title("HR Policy Assistant")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["💬 Ask a Question", "🗂️ Admin Panel"])
st.sidebar.markdown("---")
st.sidebar.caption("Powered by Groq LLaMA 3.3 · LangChain · FAISS")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — CHAT
# ═══════════════════════════════════════════════════════════════════════════
if page == "💬 Ask a Question":
    st.title("📘 HR Policy Assistant")
    st.write("The HR Policy Assistant is a RAG-powered AI chatbot that enables employees to ask natural language questions about company policies, benefits, and leave rules and receive accurate, citation-backed answers from official HR documents. Users can access the live MVP, navigate to “Ask a Question,” and submit queries such as “How many sick leaves are allowed?” The system is preloaded with sample documents like the Employee Handbook, Leave Policy, and Benefits Policy. No login is required for testing, and admins can upload additional PDFs through the Admin Panel.")
    st.divider()

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📌 Sources", expanded=False):
                    for s in msg["sources"]:
                        st.markdown(f"- **{s['document']}** — Page {s['page']}")

    # Chat input
    question = st.chat_input("e.g. How many sick leaves am I entitled to?")

    if question:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Call API
        with st.chat_message("assistant"):
            with st.spinner("Searching policies..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/ask",
                        json={"question": question},
                        timeout=60
                    )
                    if resp.status_code != 200:
                        answer  = "⚠️ Backend error. Please try again."
                        sources = []
                    else:
                        data    = resp.json()
                        answer  = data.get("answer", "").strip()
                        sources = data.get("sources", [])

                        if not answer:
                            answer = "🤷 I don't know based on the available policy documents."

                except requests.exceptions.RequestException:
                    answer  = "❌ Could not connect to backend API. Make sure `uvicorn api:app` is running."
                    sources = []

            st.markdown(answer)
            if sources:
                with st.expander("📌 Sources", expanded=True):
                    for s in sources:
                        st.markdown(f"- **{s['document']}** — Page {s['page']}")

        # Save assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — ADMIN PANEL
# ═══════════════════════════════════════════════════════════════════════════
else:
    st.title("🗂️ Admin Panel — Document Management")
    st.write("Upload new HR policy PDFs to expand the knowledge base.")
    st.divider()

    # ── Indexed documents list ──────────────────────────────────────────────
    st.subheader("📚 Currently Indexed Documents")
    try:
        r = requests.get(f"{API_URL}/admin/documents", timeout=10)
        if r.status_code == 200:
            docs = r.json().get("documents", [])
            if docs:
                for d in docs:
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        st.write("📄")
                    with col2:
                        st.write(f"**{d['name']}**  `{d['filename']}`")
            else:
                st.info("No documents indexed yet.")
            st.success(f"Total: **{r.json().get('count', 0)}** documents indexed")
        else:
            st.error("Could not fetch document list from API.")
    except requests.exceptions.RequestException:
        st.error("❌ Cannot reach backend API.")

    st.divider()

    # ── Upload new PDF ──────────────────────────────────────────────────────
    st.subheader("⬆️ Upload a New Policy Document")
    uploaded = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="The document will be automatically indexed into the knowledge base."
    )

    if uploaded:
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("📤 Upload & Index"):
                with st.spinner(f"Uploading **{uploaded.name}**..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/admin/upload",
                            files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                            timeout=30
                        )
                        if resp.status_code == 200:
                            result = resp.json()
                            if result.get("status") == "success":
                                st.success(f"✅ {result['message']}")
                                st.info("ℹ️ Indexing runs in the background. Refresh in ~10 seconds.")
                                st.rerun()
                            else:
                                st.error(result.get("message", "Upload failed."))
                        else:
                            st.error("Upload failed. Check the backend logs.")
                    except requests.exceptions.RequestException:
                        st.error("❌ Cannot reach backend API.")

    st.divider()
    st.subheader("💡 Tips")
    st.markdown("""
- PDF files are automatically split into chunks and embedded for semantic search.
- After uploading, wait ~10 seconds before querying the new document.
- Supported format: **PDF only**.
- Large files (50+ pages) may take up to 30 seconds to index.
    """)