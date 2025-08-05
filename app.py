import streamlit as st



import streamlit as st
from agent_system import WeatherRAGAgent, PDFManager, check_environment
from web_search import web_search

# Initialize agent and PDF manager
@st.cache_resource
def load_components():
    check_environment()
    agent = WeatherRAGAgent()
    pdf_manager = PDFManager()
    return agent, pdf_manager

agent, pdf_manager = load_components()

# Streamlit UI
st.set_page_config(page_title="Weather & Document Assistant", layout="centered")
st.title("🌤️📄 Weather & Document Assistant")

# Section: PDF Upload
st.subheader("📄 Upload a PDF for Document Q&A")
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        try:
            # Save the file temporarily
            with open(f"uploaded_pdfs/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.read())
            num_chunks = pdf_manager.process_and_store_pdf(f"uploaded_pdfs/{uploaded_file.name}", doc_name=uploaded_file.name)
            st.success(f"✅ PDF processed and indexed successfully with {num_chunks} chunks.")
        except Exception as e:
            st.error(f"❌ Failed to process PDF: {e}")

# Section: Query Input
st.subheader("💬 Ask a Question")
query = st.text_input("Ask about weather, uploaded PDF content, or search the web:")

# Web search toggle
enable_web_search = st.checkbox("🔍 Enable Web Search")

if st.button("Submit Query") and query:
    with st.spinner("Processing..."):
        if enable_web_search:
            # Show web search results
            web_results = web_search(query)
            st.markdown("### 🔍 Web Search Results:")
            st.markdown(web_results)
            
            # Also show agent response
            # st.markdown("### 🤖 AI Assistant Response:")
            # agent_response = agent.process_query(query)
            # st.write(agent_response)
        else:
            # Regular agent processing
            response = agent.process_query(query)
            st.markdown("### 🤖 Response:")
            st.write(response)