import streamlit as st
from retrieve import retrieve
from transformers import pipeline

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(
    page_title="OneLore | The Pirate Archives",
    page_icon="☠️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# 2. Custom CSS
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stChatMessage {
        background-color: #1E232B;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid #30363D;
    }

    .streamlit-expanderHeader {
        background-color: #161B22;
        color: #FFD700;
        font-weight: 600;
        border-radius: 5px;
    }
    
    .source-box {
        background-color: #0d1117;
        border-left: 4px solid #FFD700;
        padding: 12px;
        margin-top: 8px;
        border-radius: 0 5px 5px 0;
        font-size: 0.9em;
        color: #c9d1d9;
    }

    h1 {
        background: -webkit-linear-gradient(45deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 3. Load Model
# -----------------------------
@st.cache_resource
def load_model():
    with st.spinner("Decoding Poneglyphs..."):
        return pipeline("question-answering", model="deepset/roberta-base-squad2")

qa_model = load_model()

# -----------------------------
# 4. Sidebar UI
# -----------------------------
with st.sidebar:
    st.title("🏴‍☠️ OneLore")
    st.markdown("---")
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -----------------------------
# 5. Main Chat UI
# -----------------------------
col1, col2 = st.columns([1, 5])
with col1:
    # Using a standard emoji instead of a URL to prevent network image errors
    st.markdown("# 🏴‍☠️") 
with col2:
    st.title("OneLore")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Whatchu wanna know about today?", "sources": []}
    ]

# Display Chat History
for msg in st.session_state.messages:
    # FIX: Removed the 'avatar' parameter entirely to use defaults
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if "sources" in msg and msg["sources"]:
            with st.expander("📜 View Logbook Entries (Sources)"):
                for s in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-box">
                        <strong style="color: #FFD700;">{s['title']}</strong> <em style="color: #8b949e;">({s['category']})</em><br>
                        {s['text']}
                    </div>
                    """, unsafe_allow_html=True)

# -----------------------------
# 6. Input & Processing
# -----------------------------
user_input = st.chat_input("Enter your query here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # FIX: Removed 'avatar' parameter here too
    with st.chat_message("user"):
        st.markdown(user_input)

    retrieved = retrieve(user_input, top_k=3)
    
    if not retrieved:
        response = "My logs contain no records of this matter."
        sources = []
    else:
        context = "\n\n".join([r["text"] for r in retrieved])
        try:
            result = qa_model(question=user_input, context=context)
            response = result["answer"]
            
            if result["score"] < 0.1 or not response.strip():
                response = "I found related logs, but they don't seem to contain a direct answer."
        except:
            response = "The Poneglyphs are undecipherable at this moment."

        sources = [{
            "text": r["text"],
            "title": r["metadata"]["title"],
            "category": r["metadata"]["category"]
        } for r in retrieved]

    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "sources": sources
    })
    
    st.rerun()