import streamlit as st
import openai
from dotenv import load_dotenv
import os
import time
import json
import requests
import tempfile
import logging
from typing import List, Dict, Optional

# Import LangChain components
from langchain_aws import ChatBedrock
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.7
MAX_RETRIES = 3
RETRY_DELAY = 1
FAISS_INDEX_NAME = "faiss_index"

# Load environment variables
load_dotenv()

class ChatInterface:
    def __init__(self):
        self._initialize_session_state()
        self._setup_llm_clients()
        self._setup_ui()
        
    def _initialize_session_state(self):
        """Initialize required session state variables."""
        if "logged_in" not in st.session_state or not st.session_state.logged_in:
            st.warning("Access Denied: Please log in to view the Chat Interface.")
            st.stop()
            
        default_state = {
            "messages": [],
            "openai_available": False,
            "bedrock_available": False,
            "bedrock_model_name": None,
            "selected_language": "English",
            "selected_llm_provider": None,
            "system_prompt_override": "",
            "app_max_tokens": DEFAULT_MAX_TOKENS,
            "app_temperature": DEFAULT_TEMPERATURE
        }
        
        for key, value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _setup_llm_clients(self):
        """Initialize LLM clients based on available credentials."""
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        self.AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.AWS_REGION_NAME = os.getenv("AWS_REGION_NAME", "us-east-1")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        
        self.llm_openai_client = None
        self.llm_bedrock_instance = None
        self.llm_gemini_available = bool(self.GEMINI_API_KEY)
        
        self._init_openai_client()
        self._init_bedrock_client()
        
    def _init_openai_client(self):
        """Initialize OpenAI client if API key is available."""
        if not self.OPENAI_API_KEY:
            st.session_state.openai_available = False
            return
            
        try:
            self.llm_openai_client = openai.OpenAI(api_key=self.OPENAI_API_KEY)
            st.session_state.openai_available = True
        except Exception as e:
            logger.error(f"OpenAI initialization error: {e}")
            st.session_state.openai_available = False
            st.warning(f"Could not initialize OpenAI client: {e}")

    def _init_bedrock_client(self):
        """Initialize Bedrock client if AWS credentials are available."""
        if not (self.AWS_ACCESS_KEY_ID and self.AWS_SECRET_ACCESS_KEY):
            st.session_state.bedrock_available = False
            return
            
        try:
            self.llm_bedrock_instance = ChatBedrock(
                credentials_profile_name=None,
                region_name=self.AWS_REGION_NAME,
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                streaming=True
            )
            st.session_state.bedrock_available = True
            st.session_state.bedrock_model_name = "Amazon Bedrock (Claude Sonnet)"
        except Exception as e:
            logger.error(f"Bedrock initialization error: {e}")
            st.session_state.bedrock_available = False
            st.warning(f"Could not initialize Bedrock client: {e}")

    def _setup_ui(self):
        """Configure the Streamlit UI layout and styling."""
        self._apply_custom_styles()
        self._setup_page_config()
        self._setup_sidebar()
        self._setup_chat_interface()
        
    def _apply_custom_styles(self):
        """Apply custom CSS styles for the application."""
        st.markdown("""
        <style>
        /* Your existing CSS styles here */
        </style>
        """, unsafe_allow_html=True)
        
    def _setup_page_config(self):
        """Configure the page settings."""
        st.set_page_config(
            page_title="Job Prep Pro Chat",
            page_icon="💬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title("Job Prep Pro: AI Recruitment Consultant")

    def _setup_sidebar(self):
        """Configure the sidebar settings and controls."""
        with st.sidebar:
            st.header("Settings")
            
            # Language selection
            language_options = ["English", "日本語 (Japanese)"]
            current_lang_idx = 0 if st.session_state.selected_language == "English" else 1
            st.session_state.selected_language = st.selectbox(
                "Select Language:",
                language_options,
                index=current_lang_idx,
                key="sidebar_language_selector"
            )
            
            # LLM provider selection
            available_llms = self._get_available_llms()
            if not available_llms:
                st.error("No LLM providers configured. Check .env.")
                st.stop()
                
            if "selected_llm_provider" not in st.session_state:
                st.session_state.selected_llm_provider = available_llms[0]
                
            st.session_state.selected_llm_provider = st.selectbox(
                "Select LLM Provider:",
                available_llms,
                key="sidebar_llm_provider_selector"
            )
            
            st.markdown("---")
            if st.button("Start New Chat 🔄", key="new_chat_button"):
                self._reset_chat()
                
    def _get_available_llms(self) -> List[str]:
        """Get list of available LLM providers."""
        available_llms = []
        if st.session_state.openai_available:
            available_llms.append("OpenAI (GPT-3.5 Turbo)")
        if st.session_state.bedrock_available:
            available_llms.append(st.session_state.bedrock_model_name)
        if self.llm_gemini_available:
            available_llms.append("Google Gemini (gemini-pro)")
        return available_llms
        
    def _reset_chat(self):
        """Reset the chat history."""
        st.session_state.messages = []
        initial_msg = self._get_initial_assistant_message()
        st.session_state.messages.append({"role": "assistant", "content": initial_msg})
        st.rerun()
        
    def _get_initial_assistant_message(self) -> str:
        """Get the initial assistant message based on selected language."""
        if st.session_state.selected_language == "English":
            return "Hello there! I am Job Prep Pro, your AI recruitment consultant. I am here to help you ace your next interview. Please paste the job description you'd like to practice for, and we'll get started."
        return "こんにちは！私はAI採用コンサルタントのJob Prep Proです。次の面接を成功させるお手伝いをします。練習したい職務記述書を貼り付けてください。すぐに始めましょう。"

    def _setup_chat_interface(self):
        """Configure the main chat interface."""
        self._display_chat_messages()
        
        # Initialize chat if empty
        if not st.session_state.messages:
            initial_msg = self._get_initial_assistant_message()
            st.session_state.messages.append({"role": "assistant", "content": initial_msg})
            
        # Handle user input
        prompt = st.chat_input(self._get_chat_placeholder())
        if prompt:
            self._handle_user_message(prompt)
            
    def _get_chat_placeholder(self) -> str:
        """Get the chat input placeholder based on selected language."""
        return "Type your message here..." if st.session_state.selected_language == "English" else "ここにメッセージを入力してください..."
        
    def _display_chat_messages(self):
        """Display all chat messages from history."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
    def _handle_user_message(self, prompt: str):
        """Process a user message and generate a response."""
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner(self._get_thinking_message()):
                full_response = self._generate_response(prompt)
                
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
    def _get_thinking_message(self) -> str:
        """Get the thinking message based on selected language."""
        return "Job Prep Pro is thinking..." if st.session_state.selected_language == "English" else "Job Prep Proが考えています..."
        
    def _generate_response(self, prompt: str) -> str:
        """Generate a response to the user's prompt."""
        try:
            # Retrieve relevant context
            retrieved_context = self._retrieve_context(prompt)
            
            # Prepare messages for LLM
            messages = self._prepare_messages(prompt, retrieved_context)
            
            # Get response based on selected provider
            provider = st.session_state.selected_llm_provider
            if provider == "OpenAI (GPT-3.5 Turbo)":
                return self._get_openai_response(messages)
            elif provider == st.session_state.bedrock_model_name:
                return self._get_bedrock_response(messages)
            elif provider == "Google Gemini (gemini-pro)":
                return self._get_gemini_response(messages, prompt, retrieved_context)
            else:
                return self._get_error_message("No valid LLM provider selected")
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_error_message(str(e))
            
    def _retrieve_context(self, prompt: str) -> str:
        """Retrieve relevant context from the vector store."""
        if not hasattr(self, "faiss_vectorstore"):
            self.faiss_vectorstore = self._get_faiss_index()
            
        if not self.faiss_vectorstore:
            return ""
            
        try:
            with st.spinner("Retrieving relevant information..."):
                docs = self.faiss_vectorstore.similarity_search(prompt, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])
                if context:
                    st.info("Found relevant context from uploaded documents.")
                return context
        except Exception as e:
            logger.warning(f"Document retrieval error: {e}")
            return ""
            
    @st.cache_resource(ttl=3600)
    def _get_faiss_index(_self):  # Changed parameter name to _self
        """Load or create the FAISS vector store."""
        data_dir = "rag_data"
        index_path = os.path.join(data_dir, FAISS_INDEX_NAME)
        
        if not os.path.exists(index_path):
            return None
            
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=_self.OPENAI_API_KEY)  # Access through _self
            return FAISS.load_local(
                index_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return None
            
    def _prepare_messages(self, prompt: str, context: str) -> List[Dict]:
        """Prepare messages for the LLM including system prompt and context."""
        system_prompt = self._get_system_prompt()
        messages = [SystemMessage(content=system_prompt)]
        
        if context:
            messages.append(SystemMessage(content=f"--- Relevant Context ---\n{context}\n---"))
            
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
                
        return messages
        
    def _get_system_prompt(self) -> str:
        """Get the system prompt based on selected language."""
        if st.session_state.selected_language == "English":
            return """
            [Your English system prompt content here]
            """
        return """
        [Your Japanese system prompt content here]
        """
        
    def _get_openai_response(self, messages: List[Dict]) -> str:
        """Get response from OpenAI API."""
        if not self.llm_openai_client:
            return self._get_error_message("OpenAI client not initialized")
            
        try:
            openai_messages = [{"role": m.type, "content": m.content} for m in messages]
            response = self.llm_openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=openai_messages,
                temperature=st.session_state.app_temperature,
                max_tokens=st.session_state.app_max_tokens,
                stream=True
            )
            
            return self._stream_response(response)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._get_error_message("OpenAI API Error")
            
    def _get_bedrock_response(self, messages: List[Dict]) -> str:
        """Get response from Bedrock API."""
        if not self.llm_bedrock_instance:
            return self._get_error_message("Bedrock client not initialized")
            
        try:
            response = self.llm_bedrock_instance.stream(messages)
            return self._stream_response(response)
        except Exception as e:
            logger.error(f"Bedrock API error: {e}")
            return self._get_error_message("Bedrock API Error")
            
    def _get_gemini_response(self, messages: List[Dict], prompt: str, context: str) -> str:
        """Get response from Gemini API."""
        if not self.GEMINI_API_KEY:
            return self._get_error_message("Gemini API key not configured")
            
        try:
            # Prepare messages in Gemini format
            gemini_messages = []
            combined_prompt = self._get_system_prompt()
            
            if context:
                combined_prompt += f"\n\n--- Relevant Context ---\n{context}\n---"
                
            gemini_messages.append({
                "role": "user",
                "parts": [{"text": f"{combined_prompt}\n\n{prompt}"}]
            })
            
            payload = {
                "contents": gemini_messages,
                "generationConfig": {
                    "temperature": st.session_state.app_temperature,
                    "maxOutputTokens": st.session_state.app_max_tokens,
                }
            }
            
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.GEMINI_API_KEY}"
            
            for attempt in range(MAX_RETRIES):
                try:
                    response = requests.post(
                        api_url,
                        headers={'Content-Type': 'application/json'},
                        data=json.dumps(payload),
                        stream=True
                    )
                    response.raise_for_status()
                    return self._process_gemini_stream(response)
                except requests.exceptions.RequestException as e:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY * (attempt + 1))
                        continue
                    raise e
                    
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._get_error_message("Gemini API Error")
            
    def _process_gemini_stream(self, response) -> str:
        """Process streaming response from Gemini API."""
        response_container = st.empty()
        full_response = ""
        
        for chunk_line in response.iter_lines():
            if chunk_line:
                decoded_line = chunk_line.decode('utf-8')
                if decoded_line.startswith("data:"):
                    try:
                        json_data = json.loads(decoded_line[len("data:"):])
                        if json_data.get('candidates') and json_data['candidates'][0].get('content') and json_data['candidates'][0]['content'].get('parts'):
                            text_chunk = json_data['candidates'][0]['content']['parts'][0].get('text', '')
                            full_response += text_chunk
                            response_container.markdown(full_response + "▌")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse Gemini response: {e}")
                        
        response_container.markdown(full_response)
        return full_response
        
    def _stream_response(self, response) -> str:
        """Stream the response from the LLM."""
        response_container = st.empty()
        full_response = ""
        
        for chunk in response:
            if hasattr(chunk, "choices") and chunk.choices[0].delta.content:  # OpenAI format
                content = chunk.choices[0].delta.content
            elif hasattr(chunk, "content"):  # Bedrock format
                content = chunk.content
            else:
                continue
                
            full_response += content
            response_container.markdown(full_response + "▌")
            
        response_container.markdown(full_response)
        return full_response
        
    def _get_error_message(self, error: str) -> str:
        """Get user-friendly error message based on selected language."""
        if st.session_state.selected_language == "English":
            return f"I'm sorry, I encountered an error: {error}. Please try again later."
        return f"申し訳ありませんが、エラーが発生しました: {error}。後でもう一度お試しください。"

# Run the chat interface
if __name__ == "__main__":
    chat_interface = ChatInterface()