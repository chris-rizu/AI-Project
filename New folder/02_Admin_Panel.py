import streamlit as st
import os
import tempfile
import time
import shutil
from typing import List, Optional
import logging

# Import LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "rag_data"
FAISS_INDEX_NAME = "faiss_index"
FAISS_INDEX_PATH = os.path.join(DATA_DIR, FAISS_INDEX_NAME)
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.7

class AdminConsole:
    def __init__(self):
        self._check_authentication()
        self._setup_ui()
        
    def _check_authentication(self):
        """Verify user is logged in before showing admin console."""
        if "logged_in" not in st.session_state or not st.session_state.logged_in:
            st.warning("Access Denied: Please log in to view the Admin Panel.")
            st.stop()
            
    def _setup_ui(self):
        """Configure the Streamlit UI for admin console."""
        self._apply_custom_styles()
        self._setup_page_config()
        self._display_rag_management()
        self._display_llm_configuration()
        self._display_app_settings()
        self._display_user_management()
        
    def _apply_custom_styles(self):
        """Apply custom CSS styles."""
        st.markdown("""
        <style>
        /* Your existing CSS styles here */
        </style>
        """, unsafe_allow_html=True)
        
    def _setup_page_config(self):
        """Configure page settings."""
        st.title("Admin Panel")
        st.markdown("---")
        st.info("This is the secure admin console for Job Prep Pro. Only authorized users can see this.")
        
    def _display_rag_management(self):
        """Display the RAG data management section."""
        st.header("1. RAG Data Management")
        st.write("Upload client documents (PDF, TXT) to enhance the AI's knowledge base for detailed answers.")
        
        self._handle_file_upload()
        self._display_uploaded_files()
        self._display_faiss_status()
        st.markdown("---")
        
    def _handle_file_upload(self):
        """Handle document upload and processing."""
        uploaded_file = st.file_uploader(
            "Upload Document", 
            type=["pdf", "txt"], 
            key="rag_file_uploader"
        )
        
        if not uploaded_file:
            return
            
        # Save to temporary file first
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
            
        # Define permanent path
        perm_path = os.path.join(DATA_DIR, uploaded_file.name)
        
        # Check for existing file
        if os.path.exists(perm_path):
            st.warning(f"File '{uploaded_file.name}' already exists.")
            os.remove(temp_path)
            return
            
        # Move to permanent location and process
        try:
            os.rename(temp_path, perm_path)
            if self._process_document(perm_path, uploaded_file.name):
                st.success(f"File '{uploaded_file.name}' processed successfully.")
                st.rerun()
        except Exception as e:
            logger.error(f"File processing error: {e}")
            st.error(f"Error processing file: {e}")
            
    def _process_document(self, file_path: str, file_name: str) -> bool:
        """Process and index a document."""
        try:
            # Determine loader based on file type
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                st.error(f"Unsupported file type: {file_name}")
                return False
                
            # Load and split document
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create or update FAISS index
            os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
            
            if self._faiss_index_exists():
                vectorstore = FAISS.load_local(
                    FAISS_INDEX_PATH, 
                    self._get_embeddings(), 
                    allow_dangerous_deserialization=True
                )
                vectorstore.add_documents(chunks)
            else:
                vectorstore = FAISS.from_documents(chunks, self._get_embeddings())
                
            vectorstore.save_local(FAISS_INDEX_PATH)
            return True
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            st.error(f"Error processing {file_name}: {e}")
            return False
            
    @st.cache_resource
    def _get_embeddings(self):
        """Get the OpenAI embeddings model."""
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            st.error("OpenAI API key required for embeddings.")
            st.stop()
        return OpenAIEmbeddings(openai_api_key=openai_key)
        
    def _faiss_index_exists(self) -> bool:
        """Check if FAISS index exists."""
        return os.path.exists(FAISS_INDEX_PATH) and \
               os.path.isfile(os.path.join(FAISS_INDEX_PATH, "index.faiss")) and \
               os.path.isfile(os.path.join(FAISS_INDEX_PATH, "index.pkl"))
               
    def _display_uploaded_files(self):
        """Display uploaded files with delete option."""
        st.subheader("Uploaded Documents")
        uploaded_files = self._get_uploaded_files()
        
        if not uploaded_files:
            st.info("No documents uploaded yet.")
            return
            
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_file = st.selectbox(
                "Select document to delete:", 
                [""] + uploaded_files,
                key="delete_doc_selector"
            )
        with col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Delete Selected", key="delete_doc_button"):
                if selected_file and self._delete_document(selected_file):
                    st.rerun()
                    
    def _get_uploaded_files(self) -> List[str]:
        """Get list of uploaded files."""
        if not os.path.exists(DATA_DIR):
            return []
            
        return sorted([
            f for f in os.listdir(DATA_DIR) 
            if os.path.isfile(os.path.join(DATA_DIR, f)) and 
               not f.startswith("index.") and 
               f != FAISS_INDEX_NAME
        ])
        
    def _delete_document(self, file_name: str) -> bool:
        """Delete a document and rebuild the index."""
        file_path = os.path.join(DATA_DIR, file_name)
        
        if not os.path.exists(file_path):
            st.warning(f"File not found: {file_name}")
            return False
            
        try:
            os.remove(file_path)
            self._rebuild_faiss_index()
            st.success(f"Deleted: {file_name}")
            return True
        except Exception as e:
            logger.error(f"Delete error: {e}")
            st.error(f"Error deleting {file_name}: {e}")
            return False
            
    def _rebuild_faiss_index(self):
        """Rebuild the FAISS index from remaining documents."""
        st.info("Rebuilding FAISS index...")
        
        # Clear existing index
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        
        # Process all remaining documents
        all_docs = []
        for file_name in self._get_uploaded_files():
            file_path = os.path.join(DATA_DIR, file_name)
            
            try:
                if file_path.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif file_path.endswith(".txt"):
                    loader = TextLoader(file_path)
                else:
                    continue
                    
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Skipping {file_name}: {e}")
                
        if all_docs:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(all_docs)
            vectorstore = FAISS.from_documents(chunks, self._get_embeddings())
            vectorstore.save_local(FAISS_INDEX_PATH)
            st.success("Index rebuilt successfully.")
        else:
            st.info("No documents remaining, index cleared.")
            
    def _display_faiss_status(self):
        """Display FAISS index status for debugging."""
        st.subheader("FAISS Index Status (Debug)")
        exists = self._faiss_index_exists()
        st.write(f"FAISS Index Exists: {exists}")
        
        if exists:
            try:
                files = os.listdir(FAISS_INDEX_PATH)
                st.write(f"Index Files: {files}")
                if "index.faiss" in files and "index.pkl" in files:
                    st.success("Core index files present")
                else:
                    st.warning("Missing core index files")
            except Exception as e:
                st.error(f"Error listing index: {e}")
                
    def _display_llm_configuration(self):
        """Display LLM configuration section."""
        st.header("2. LLM Provider Configuration")
        st.write("Manage API keys and model settings for different LLM providers.")
        
        with st.form("llm_config_form"):
            self._display_openai_settings()
            self._display_bedrock_settings()
            self._display_gemini_settings()
            
            if st.form_submit_button("Save LLM Settings"):
                st.success("Settings updated in session. Update .env for persistence.")
        st.markdown("---")
        
    def _display_openai_settings(self):
        """Display OpenAI configuration options."""
        st.subheader("OpenAI Settings")
        st.text_input(
            "OpenAI API Key", 
            type="password", 
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Used for both chat and embeddings."
        )
        
    def _display_bedrock_settings(self):
        """Display Bedrock configuration options."""
        st.subheader("Amazon Bedrock Settings")
        st.text_input(
            "AWS Access Key ID", 
            type="password", 
            value=os.getenv("AWS_ACCESS_KEY_ID", "")
        )
        st.text_input(
            "AWS Secret Access Key", 
            type="password", 
            value=os.getenv("AWS_SECRET_ACCESS_KEY", "")
        )
        st.text_input(
            "AWS Region Name", 
            value=os.getenv("AWS_REGION_NAME", "us-east-1")
        )
        
    def _display_gemini_settings(self):
        """Display Gemini configuration options."""
        st.subheader("Google Gemini Settings")
        st.text_input(
            "Gemini API Key", 
            type="password", 
            value=os.getenv("GEMINI_API_KEY", "")
        )
        
    def _display_app_settings(self):
        """Display application settings section."""
        st.header("3. Application Settings")
        st.write("General settings for the Job Prep Pro application's behavior.")
        
        with st.form("app_settings_form"):
            # Initialize session state settings if not present
            if "app_max_tokens" not in st.session_state:
                st.session_state.app_max_tokens = DEFAULT_MAX_TOKENS
            if "app_temperature" not in st.session_state:
                st.session_state.app_temperature = DEFAULT_TEMPERATURE
            if "app_system_prompt" not in st.session_state:
                st.session_state.app_system_prompt = ""
                
            # Display settings controls
            st.number_input(
                "Max Tokens", 
                min_value=100, 
                max_value=4000, 
                value=st.session_state.app_max_tokens,
                step=100
            )
            
            st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=st.session_state.app_temperature,
                step=0.05
            )
            
            st.text_area(
                "System Prompt Override", 
                value=st.session_state.app_system_prompt,
                height=200
            )
            
            if st.form_submit_button("Save Application Settings"):
                st.success("Settings saved for current session.")
                st.rerun()
        st.markdown("---")
        
    def _display_user_management(self):
        """Display user management section (placeholder)."""
        st.header("4. User Management")
        st.write("Manage application users and their roles. Requires backend database.")
        
        with st.form("user_management_form"):
            st.text_input("New Username", key="new_user_username")
            st.text_input("New Password", type="password", key="new_user_password")
            
            if st.form_submit_button("Add User"):
                st.warning("User management requires database integration.")
                
        st.subheader("Current Users (Placeholder)")
        st.write("Future: List of users with edit/delete options")
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

# Run the admin console
if __name__ == "__main__":
    admin_console = AdminConsole()