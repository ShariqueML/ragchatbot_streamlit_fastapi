import streamlit as st
import requests
import uuid
from typing import List, Optional, Dict, Any, TypedDict
import hashlib
# --- 1. Configuration ---
API_URL = "http://localhost:8000"

# --- 2. Streamlit UI Components and State Management ---
st.set_page_config(page_title="Conversational RAG Chatbot", layout="wide")
st.title("💬 Conversational RAG Chatbot")
st.caption("Powered by a FastAPI backend")

# Initialize session state for conversations, messages, and the current session ID
if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever_ready" not in st.session_state:
    st.session_state.retriever_ready = False
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}
# New state variable to handle negative feedback comments
if "negative_feedback_for" not in st.session_state:
    st.session_state.negative_feedback_for = None

# Initialize session state for storing uploaded file hashes
if 'uploaded_file_hashes' not in st.session_state:
    st.session_state.uploaded_file_hashes = set()
if 'uploaded_files_info' not in st.session_state:
    st.session_state.uploaded_files_info = []

def get_file_hash(file):
    """Generates a unique hash for a file using its name, size, and content."""
    hasher = hashlib.sha256()
    # Read a small chunk of the file to ensure content-based uniqueness
    # Combine with file name and size for a robust identifier
    file_content = file.getvalue()
    hasher.update(file.name.encode('utf-8'))
    hasher.update(str(file.size).encode('utf-8'))
    hasher.update(file_content[:1024])  # Use first 1KB of content
    return hasher.hexdigest()
# --- 3. Helper Functions for Backend Communication ---
def send_documents_to_backend(uploaded_files):
    """Sends uploaded files to the FastAPI backend for processing."""
    files = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
    try:
        response = requests.post(f"{API_URL}/upload-documents", files=files, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error processing documents: {e}")
        return None

def send_chat_message_to_backend(prompt: str, chat_history: List[Dict[str, Any]]):
    """Sends a chat message to the FastAPI backend and handles the response."""
    history_for_api = [
        {"role": msg.get("role"), "content": msg.get("content")} 
        for msg in chat_history
    ]
    
    payload = {
        "user_question": prompt,
        "session_id": st.session_state.session_id,
        "chat_history": history_for_api
    }
    
    try:
        response = requests.post(f"{API_URL}/chat", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with the backend: {e}")
        return None

def send_feedback_to_backend(telemetry_entry_id: str, feedback_score: int, feedback_text: Optional[str] = None):
    """Sends feedback to the FastAPI backend."""
    payload = {
        "session_id": st.session_state.session_id,
        "telemetry_entry_id": telemetry_entry_id,
        "feedback_score": feedback_score,
        "feedback_text": feedback_text
    }
    try:
        response = requests.post(f"{API_URL}/feedback", json=payload)
        response.raise_for_status()
        st.toast("Feedback submitted! Thank you.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error submitting feedback: {e}")

def get_conversations_from_backend():
    """Fetches a list of all conversations from the backend."""
    try:
        response = requests.get(f"{API_URL}/conversations")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Error fetching conversations: {e}")
        return []

def get_conversation_history_from_backend(session_id: str):
    """Fetches the messages for a specific conversation ID."""
    try:
        response = requests.get(f"{API_URL}/conversations/{session_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading conversation history: {e}")
        return None

def handle_positive_feedback(telemetry_id):
    """Handles positive feedback submission."""
    send_feedback_to_backend(telemetry_id, 1)
    st.session_state.feedback_given[telemetry_id] = True


def handle_negative_feedback_comment_submit(telemetry_id, comment_text):
    """Handles the negative feedback comment submission."""
    send_feedback_to_backend(telemetry_id, -1, comment_text)
    st.session_state.feedback_given[telemetry_id] = True
    st.session_state.negative_feedback_for = None


def refresh_conversations():
    """Refreshes the conversation list in the sidebar."""
    st.session_state.conversations = get_conversations_from_backend()

# --- 4. Sidebar for Document Upload and Conversation History ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload Text, PDF, Docx files:",
        type=["txt","pdf","docx"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    if st.button("Process Documents", key="process_docs_button"):
        if uploaded_files:
            new_uploaded_files = []
            newly_added_files_info = []
            for file in uploaded_files:
                file_hash = get_file_hash(file)

                if file_hash not in st.session_state.uploaded_file_hashes:
                    st.session_state.uploaded_file_hashes.add(file_hash)
                    new_uploaded_files.append(file)
                    newly_added_files_info.append({"name": file.name, "size": file.size})
                else:
                    st.warning(f"File '{file.name}' has already been uploaded.")

            if new_uploaded_files:
                st.session_state.uploaded_files_info.extend(newly_added_files_info)
                with st.spinner("Processing documents..."):
                    response_data = send_documents_to_backend(new_uploaded_files)
                    if response_data:
                        st.session_state.retriever_ready = True
                        st.success(response_data.get("message", "Documents processed and knowledge base ready!"))
                        st.session_state.messages = []
                        refresh_conversations()
                    else:
                        st.session_state.retriever_ready = False
                        st.error("Failed to process documents.")
        else:
            st.warning("Please upload at least one document to process.")

    st.markdown("---")
    st.header("Conversations")
    if st.button("➕ New Chat", key="new_chat_button", use_container_width=True, type="primary"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.feedback_given = {}
        st.session_state.negative_feedback_for = None
        refresh_conversations()
        st.rerun()
    
    refresh_conversations()
    
    if st.session_state.conversations:
        for conv in st.session_state.conversations:
            if st.button(
                conv["title"], 
                key=f"conv_{conv['session_id']}",
                use_container_width=True
            ):
                if st.session_state.session_id != conv["session_id"]:
                    st.session_state.session_id = conv["session_id"]
                    history = get_conversation_history_from_backend(conv["session_id"])
                    if history:
                        st.session_state.messages = history
                        st.session_state.feedback_given = {msg.get("telemetry_id"): True for msg in history if msg.get("telemetry_id")}
                    else:
                        st.session_state.messages = []
                    st.session_state.negative_feedback_for = None
                    st.rerun()

# --- 5. Main Chat Interface ---
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    # Display feedback buttons for the last AI response
    if message["role"] == "assistant" and message.get("telemetry_id") and not st.session_state.feedback_given.get(message["telemetry_id"], False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍", key=f"positive_{message['telemetry_id']}", on_click=handle_positive_feedback, args=(message['telemetry_id'],)):
                pass
        with col2:
            if st.button("👎", key=f"negative_{message['telemetry_id']}"):
                st.session_state.negative_feedback_for = message['telemetry_id']
                st.rerun()
        
        # --- NEW LOGIC FOR NEGATIVE FEEDBACK COMMENT ---
        # Only render the comment input if this is the message the user clicked thumbs down on
        if st.session_state.negative_feedback_for == message['telemetry_id']:
            with st.container():
                comment = st.text_area(
                    "Please provide some details (optional):", 
                    key=f"feedback_text_{message['telemetry_id']}"
                )
                if st.button("Submit Comment", key=f"submit_feedback_button_{message['telemetry_id']}"):
                    handle_negative_feedback_comment_submit(message['telemetry_id'], comment)

# Chat input for new questions
if st.session_state.retriever_ready:
    if prompt := st.chat_input("Ask me anything about the uploaded documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_data = send_chat_message_to_backend(prompt, st.session_state.messages)
                if response_data:
                    ai_response = response_data.get("ai_response", "Sorry, I couldn't generate a response.")
                    telemetry_id = response_data.get("telemetry_entry_id")

                    st.markdown(ai_response)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ai_response,
                        "telemetry_id": telemetry_id
                    })
                    
                    refresh_conversations()
                    
                    if telemetry_id:
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("👍", key=f"positive_{telemetry_id}", on_click=handle_positive_feedback, args=(telemetry_id,)):
                                pass
                        with col2:
                            if st.button("👎", key=f"negative_{telemetry_id}"):
                                st.session_state.negative_feedback_for = telemetry_id
                                st.rerun()
                else:
                    st.markdown("An error occurred.")
else:
    st.info("Please upload and process documents to start chatting.")