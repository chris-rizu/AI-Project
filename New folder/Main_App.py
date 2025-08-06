import streamlit as st
import os
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Get the admin password from environment variables
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# Check if admin password is set
if not ADMIN_PASSWORD:
    st.error("ADMIN_PASSWORD not set in .env file. Please configure it to secure your app.")
    st.stop() # Stop the app if key is missing

# Initialize session state for login status
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    """Displays the login form and handles authentication."""
    st.set_page_config(
        page_title="Job Prep Pro Login",
        page_icon="💼",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

        html, body {
            height: 100%; /* Ensure html and body take full height */
            margin: 0;
            padding: 0;
            overflow: hidden; /* Prevent scrollbars if not needed */
        }

        .stApp {
            background-color: #000000; /* Black background for the whole app */
            color: #f0f2f6; /* Light text for general content */
            font-family: 'Roboto', sans-serif; /* Apply Roboto font */
            display: flex; /* Use flexbox for centering */
            flex-direction: column; /* Stack children vertically */
            align-items: center; /* Center horizontally */
            justify-content: center; /* Center vertically */
            min-height: 100vh; /* Ensure it takes full viewport height */
            padding: 0; /* Remove default padding */
            margin: 0; /* Remove default margin */
        }
        .stApp > header {
            display: none; /* Hide default Streamlit header */
        }

        /* Removed .login-card styles as the card itself is removed */
        .login-title {
            color: #f0f2f6; /* White for main title */
            font-size: 3.5em; /* Larger title */
            margin-bottom: 0.1em;
            font-weight: 700; /* Bolder */
            text-align: center; /* Center the heading */
        }
        .login-subtitle {
            font-size: 1.5em;
            color: #cccccc; /* Lighter grey for subtitle */
            margin-bottom: 2em;
            text-align: center; /* Center the subtitle */
        }
        .stTextInput > div > div > input {
            background-color: #333333; /* Darker grey for input background */
            color: #f0f2f6; /* Light text for input */
            border: 1px solid #555555; /* Darker border */
            border-radius: 8px;
            padding: 12px 15px;
            font-size: 1.05em;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
            width: 100%; /* Make input full width */
            max-width: 350px; /* Limit max width for input */
            margin: 0 auto; /* Center input */
            display: block; /* Ensure it takes full width and centers */
        }
        .stTextInput label {
            font-weight: 500;
            color: #f0f2f6; /* Light color for label */
            margin-bottom: 5px;
            display: block;
            text-align: center; /* Center the label */
        }
        .stButton > button {
            background-color: #4CAF50; /* Green sign-in button */
            color: white;
            border-radius: 8px;
            padding: 12px 25px;
            font-size: 1.2em;
            font-weight: bold;
            border: none;
            cursor: pointer;
            width: 100%;
            max-width: 350px; /* Limit max width for button */
            margin-top: 20px;
            transition: background-color 0.3s ease, transform 0.2s ease;
            box-shadow: 0 4px 10px rgba(76, 175, 80, 0.3); /* Green shadow */
            display: block; /* Ensure it takes full width and centers */
            margin-left: auto;
            margin-right: auto;
        }
        .stButton > button:hover {
            background-color: #45a049; /* Darker green on hover */
            transform: translateY(-2px);
        }
        .link-text-container {
            margin-top: 20px;
            text-align: center; /* Center the links container */
            width: 100%; /* Ensure it takes full width for centering */
            max-width: 350px; /* Match input/button width */
            margin-left: auto;
            margin-right: auto;
        }
        .link-text {
            font-size: 0.95em;
            color: #88bbff; /* Lighter blue for links on dark background */
            text-decoration: none;
            cursor: pointer;
            margin: 0 10px;
            display: inline-block;
            transition: color 0.2s ease;
        }
        .link-text:hover {
            color: #6699ee; /* Darker blue on hover */
            text-decoration: underline;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Main content for the login page, directly on the black background
    st.markdown('<h1 class="login-title"> Welcome to Job Prep Pro <br>Your AI Recruitment Consultant</h1>', unsafe_allow_html=True)
    
    # Use a container to group the input and button for consistent max-width and centering
    # This container will be centered by the flexbox properties on .stApp
    with st.container():
        # Using columns to help with horizontal centering of the input/button block
        # The center column will take 2 parts of 4 (1+2+1), effectively 50% of available space
        # within the container, and the container itself is centered by .stApp flex properties.
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center: # Place input and button in the center column
            st.markdown('<label style="text-align:center; display:block; font-weight:500; color:#f0f2f6; margin-bottom:5px;">Your password</label>', unsafe_allow_html=True)
            password_input = st.text_input("", type="password", placeholder="e.g. yoursecurepassword123", key="login_password_input")
            
            if st.button("Sign in", key="login_button"):
                if password_input == ADMIN_PASSWORD:
                    st.session_state.logged_in = True
                    st.success("Logged in successfully! Redirecting...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            
            st.markdown(
                """
                <div class="link-text-container">
                    <span class="link-text">Don't have an account?</span>
                    <span class="link-text">Forgot password?</span>
                </div>
                """, unsafe_allow_html=True
            )


def main_app_content():
    """Displays the main welcome content after successful login."""
    st.set_page_config(
        page_title="Job Prep Pro",
        page_icon="💼",
        layout="wide", # Use wide layout for main app
        initial_sidebar_state="expanded" # Ensure sidebar is expanded after login
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

        .stApp {
            background-color: #1a1a1a; /* Dark background */
            color: #f0f2f6; /* Light text */
            padding-top: 0px; /* Remove default top padding */
            font-family: 'Roboto', sans-serif; /* Apply Roboto font */
        }
        h1 {
            color: #4CAF50; /* Green title */
            text-align: center;
            font-size: 3.5em; /* Larger title */
            margin-bottom: 0.2em;
        }
        .stMarkdown p {
            font-size: 1.2em;
            text-align: center;
            color: #cccccc;
        }
        .hero-section {
            padding: 60px 0; /* Slightly reduced top/bottom padding */
            text-align: center;
            background: linear-gradient(to right, #2b2b2b, #3a3a3a); /* Subtle gradient */
            border-radius: 0px; /* Remove border-radius for full width */
            margin-bottom: 40px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
            width: 100%; /* Ensure it takes full width */
            margin-left: -1rem; /* Adjust for Streamlit's default container padding */
            margin-right: -1rem;
        }
        .call-to-action {
            margin-top: 30px;
            font-size: 1.3em;
            color: #f0f2f6;
            font-weight: bold;
        }
        .stSidebar {
            background-color: #2b2b2b; /* Darker sidebar */
            color: #f0f2f6;
            font-family: 'Roboto', sans-serif; /* Apply Roboto font to sidebar */
        }
        .st-emotion-cache-1jmve6n { /* Target sidebar header */
            color: #4CAF50; /* Green for sidebar title */
        }
        .st-emotion-cache-10qj31p { /* Target sidebar links */
            color: #f0f2f6;
        }
        .st-emotion-cache-10qj31p:hover {
            color: #4CAF50;
        }
        .app-logo {
            display: block;
            margin: 0 auto 20px auto; /* Center the logo and add bottom margin */
            width: 80px; /* Adjust size as needed */
            height: 80px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1>Welcome to Job Prep Pro! 🚀</h1>", unsafe_allow_html=True)
    st.markdown("<p>Your ultimate AI-powered recruitment consultant for interview success.</p>", unsafe_allow_html=True)
    st.markdown("<p>Select a page from the sidebar to start preparing for your next career move!</p>", unsafe_allow_html=True)
    st.markdown('<div class="call-to-action">Navigate using the left sidebar.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


    st.info("💡 **Tip:** Use the sidebar on the left to switch between the 'Chat Interface' for mock interviews and the 'Admin Panel' for configurations.")

# --- Main Application Logic ---
if not st.session_state.logged_in:
    login_page() # Show login page if not logged in
else:
    main_app_content() # Show main app content (which includes the sidebar pages)
