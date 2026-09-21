import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Recipe AI Assistant 🍳", layout="wide")
st.title("🍳 Recipe RAG Assistant")

# Sidebar for PDF upload
with st.sidebar:
    st.header("📄 Recipe PDF Manager")
    uploaded_file = st.file_uploader("Upload a new recipe PDF", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Upload & Index"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            res = requests.post(f"{API_URL}/upload", files=files)
            if res.status_code == 200:
                st.success("File uploaded and indexed successfully!")
            else:
                st.error("Failed to upload file.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a recipe question (e.g., How do I make hummus?):"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(f"{API_URL}/query", json={"question": prompt})
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]

                st.markdown(answer)
                
                # Show source expander
                with st.expander("📚 View Retrieved Sources"):
                    for idx, src in enumerate(sources):
                        st.write(f"**Chunk {idx+1}:** {src}")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("Error communicating with backend.")