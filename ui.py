import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"



st.set_page_config(
    page_title="Company Policy Assistant",
    page_icon="📘",
    layout="centered"
)

st.title("📘 Company Policy Assistant")
st.write("Ask questions about company policies, leave rules, and compliance.")

st.divider()

question = st.text_input(
    "Enter your question:",
    placeholder="e.g. How many paid leaves are allowed per year?"
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching company policies..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=30
                )

                if response.status_code != 200:
                    st.error("Backend error. Please try again.")
                else:
                    data = response.json()
                    answer = data.get("answer", "").strip()
                    sources = data.get("sources", [])

                    # UI-level safety guard
                    if (
                        not answer
                        or answer.lower() in ["i don't know", "i do not know"]
                        or len(sources) == 0
                    ):
                        st.info("🤷 I don't know")
                    else:
                        st.success("Answer:")
                        st.write(answer)

                        st.divider()
                        st.caption("📌 Sources:")
                        for src in sources:
                            st.write(f"- {src}")

            except requests.exceptions.RequestException:
                st.error("Could not connect to backend API.")
