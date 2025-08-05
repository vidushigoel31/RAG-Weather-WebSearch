# agent_system.py
"""
Complete Weather & Document Assistant using LangGraph
Includes all components: weather service, PDF processing, vector store, LLM, and agent orchestration
"""

import os
import re
import requests
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from typing import TypedDict, List, Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import tempfile
import time

# Load environment variables
load_dotenv()



# from langchain.globals import set_tracer
from langchain_core.tracers import LangChainTracer
from langsmith import traceable
# Set tracer globally
# set_tracer(LangChainTracer())
import os
os.environ["LANGCHAIN_API_KEY"] = "your-api-key-here"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "weather-rag-assistant"
# from langchain_core.tracers import tracing_v2_enabled
# Data Models
class AgentState(TypedDict):
    messages: List[dict]
    query: str
    query_type: str  # "weather" or "pdf"
    weather_data: Optional[dict]
    pdf_context: Optional[str]
    final_response: Optional[str]

@dataclass
class WeatherData:
    city: str
    temperature: float
    description: str
    humidity: int
    wind_speed: float

# Weather Service
class WeatherService:
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        
        if not self.api_key:
            print("Warning: OPENWEATHER_API_KEY not found in environment variables")
    
    def get_weather(self, city: str) -> Optional[WeatherData]:
        """Fetch weather data for a given city"""
        if not self.api_key:
            print("Error: Weather API key not configured")
            return None
            
        try:
            params = {
                "q": city,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return WeatherData(
                city=data["name"],
                temperature=data["main"]["temp"],
                description=data["weather"][0]["description"],
                humidity=data["main"]["humidity"],
                wind_speed=data["wind"]["speed"]
            )
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None

# PDF Processor
class PDFProcessor:
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks for embedding"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - overlap
            
        return [chunk for chunk in chunks if chunk]

# Vector Store
class VectorStore:
    def __init__(self):
        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            chroma_dir = "./chroma_db"
            os.makedirs(chroma_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=chroma_dir)
            self.collection = self.client.get_or_create_collection(
                name="pdf_documents",
                metadata={"hnsw:space": "cosine"}
            )
            print("Vector store initialized successfully")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, texts: List[str], doc_name: str):
        """Add document chunks to vector store"""
        try:
            if not texts:
                return
                
            embeddings = self.embedding_model.encode(texts).tolist()
            
            ids = [f"{doc_name}_chunk_{i}_{int(time.time())}" for i in range(len(texts))]
            metadatas = [{"source": doc_name, "chunk_id": i} for i in range(len(texts))]
            
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added {len(texts)} chunks from {doc_name}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
    
    def search_similar(self, query: str, n_results: int = 3) -> List[str]:
        """Search for similar documents"""
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []

# LLM Service
class LLMService:
    def __init__(self):
        # Get environment variables directly
        azure_endpoint = os.getenv("AZURE_API_BASE")
        azure_api_key = os.getenv("AZURE_API_KEY")
        
        if not azure_endpoint or not azure_api_key:
            print("Warning: Azure OpenAI credentials not found in environment variables")
            print(f"AZURE_API_BASE: {azure_endpoint}")
            print(f"AZURE_API_KEY: {'*' * 10 if azure_api_key else 'None'}")
            self.llm = None
            return
            
        try:
            self.llm = AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version="2024-02-15-preview",
                deployment_name="gpt-4",  # Change this to your actual deployment name
                temperature=0.7
            )
            print("LLM service initialized successfully")
        except Exception as e:
            print(f"Error initializing LLM service: {e}")
            self.llm = None
    
    def classify_query(self, query: str) -> str:
        """Classify if query is about weather or PDF content"""
        if not self.llm:
            # Fallback classification based on keywords
            weather_keywords = ['weather', 'temperature', 'climate', 'forecast', 'rain', 'snow', 'wind', 'humidity']
            return "weather" if any(keyword in query.lower() for keyword in weather_keywords) else "pdf"
            
        try:
            system_prompt = """
            You are a query classifier. Classify the user query into one of two categories:
            - "weather": if the query is asking about weather, temperature, climate, or weather conditions
            - "pdf": if the query is asking about document content, information retrieval, or general questions
            
            Respond with only one word: either "weather" or "pdf"
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
            
            response = self.llm.invoke(messages)
            classification = response.content.strip().lower()
            
            return "weather" if "weather" in classification else "pdf"
        except Exception as e:
            print(f"Error classifying query: {e}")
            # Fallback classification based on keywords
            weather_keywords = ['weather', 'temperature', 'climate', 'forecast', 'rain', 'snow', 'wind', 'humidity']
            return "weather" if any(keyword in query.lower() for keyword in weather_keywords) else "pdf"
    
    def generate_weather_response(self, query: str, weather_data: dict) -> str:
        """Generate response for weather queries"""
        if not self.llm:
            # Fallback response without LLM
            if "error" in weather_data:
                return f"Weather Error: {weather_data['error']}"
            else:
                return f"""Weather in {weather_data.get('city', 'Unknown')}:
Temperature: {weather_data.get('temperature', 'N/A')}°C
Description: {weather_data.get('description', 'N/A')}
Humidity: {weather_data.get('humidity', 'N/A')}%
Wind Speed: {weather_data.get('wind_speed', 'N/A')} m/s"""
        
        try:
            system_prompt = """
            You are a helpful weather assistant. Use the provided weather data to answer the user's question.
            Be conversational and provide useful information based on the weather data.
            If there's an error in the weather data, acknowledge it politely and suggest alternatives.
            """
            
            if "error" in weather_data:
                weather_info = f"Error: {weather_data['error']}"
            else:
                weather_info = f"""
                Weather Data:
                City: {weather_data.get('city', 'Unknown')}
                Temperature: {weather_data.get('temperature', 'N/A')}°C
                Description: {weather_data.get('description', 'N/A')}
                Humidity: {weather_data.get('humidity', 'N/A')}%
                Wind Speed: {weather_data.get('wind_speed', 'N/A')} m/s
                """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User Query: {query}\n\n{weather_info}")
            ]
            
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Error generating weather response: {e}")
            return "I'm sorry, I'm having trouble processing your weather request right now. Please try again later."
    
    def generate_rag_response(self, query: str, context: str) -> str:
        """Generate response for PDF-based queries using RAG"""
        if not self.llm:
            # Fallback response without LLM
            if not context.strip():
                return "No relevant documents found. Please upload a PDF document first."
            else:
                return f"Based on the documents, here's the relevant context:\n\n{context[:500]}..."
        
        try:
            system_prompt = """
            You are a helpful assistant that answers questions based on the provided context.
            Use only the information from the context to answer the question.
            If the context doesn't contain relevant information, say so politely and suggest the user upload relevant documents.
            Be concise but comprehensive in your response.
            """
            
            if not context.strip():
                context = "No relevant context found in uploaded documents."
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Context: {context}\n\nQuestion: {query}")
            ]
            
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Error generating RAG response: {e}")
            return "I'm sorry, I'm having trouble processing your document query right now. Please try again later."

# Graph Nodes
class GraphNodes:
    def __init__(self):
        self.weather_service = WeatherService()
        self.vector_store = VectorStore()
        self.llm_service = LLMService()
    
    def classify_query_node(self, state: AgentState) -> AgentState:
        """Node to classify the user query"""
        query = state["query"]
        query_type = self.llm_service.classify_query(query)
        state["query_type"] = query_type
        print(f"Query classified as: {query_type}")
        return state
    
    def weather_node(self, state: AgentState) -> AgentState:
        """Node to handle weather queries"""
        query = state["query"]
        city = self._extract_city_from_query(query)
        print(f"Extracted city: {city}")
        
        if city:
            weather_data = self.weather_service.get_weather(city)
            if weather_data:
                state["weather_data"] = {
                    "city": weather_data.city,
                    "temperature": weather_data.temperature,
                    "description": weather_data.description,
                    "humidity": weather_data.humidity,
                    "wind_speed": weather_data.wind_speed
                }
            else:
                state["weather_data"] = {"error": f"Could not fetch weather data for {city}. Please check the city name."}
        else:
            state["weather_data"] = {"error": "Could not identify city in the query. Please specify a city name."}
        
        return state
    
    def pdf_rag_node(self, state: AgentState) -> AgentState:
        """Node to handle PDF-based queries using RAG"""
        query = state["query"]
        relevant_docs = self.vector_store.search_similar(query, n_results=3)
        context = "\n\n".join(relevant_docs)
        state["pdf_context"] = context
        print(f"Found {len(relevant_docs)} relevant document chunks")
        return state
    
    def generate_response_node(self, state: AgentState) -> AgentState:
        """Node to generate final response"""
        query = state["query"]
        query_type = state["query_type"]
        
        if query_type == "weather" and state.get("weather_data"):
            response = self.llm_service.generate_weather_response(query, state["weather_data"])
        elif query_type == "pdf":
            context = state.get("pdf_context", "")
            response = self.llm_service.generate_rag_response(query, context)
        else:
            response = "I'm sorry, I couldn't process your request. Please try rephrasing your question."
        
        state["final_response"] = response
        return state
    
    def _extract_city_from_query(self, query: str) -> str:
        """Extract city name from weather query"""
        # Enhanced city extraction patterns
        patterns = [
            r"weather in ([A-Za-z\s,]+)",
            r"temperature in ([A-Za-z\s,]+)",
            r"climate in ([A-Za-z\s,]+)",
            r"weather for ([A-Za-z\s,]+)",
            r"forecast for ([A-Za-z\s,]+)",
            r"weather of ([A-Za-z\s,]+)",
            r"temperature of ([A-Za-z\s,]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                city = match.group(1).strip()
                # Clean up common suffixes
                city = re.sub(r'\s+(weather|temperature|climate|forecast).*$', '', city, flags=re.IGNORECASE)
                return city
        
        # If no pattern matches, try to find city names (basic approach)
        words = query.split()
        # Look for capitalized words (potential city names)
        for word in words:
            if word[0].isupper() and len(word) > 2:
                return word
        
        return "London"  # Default fallback

# PDF Manager
class PDFManager:
    def __init__(self):
        self.processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.upload_dir = "uploaded_pdfs"
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def process_and_store_pdf(self, pdf_path: str, doc_name: str = None) -> int:
        """Process PDF and store in vector database"""
        if not doc_name:
            doc_name = os.path.basename(pdf_path)
        
        # Extract text
        text = self.processor.extract_text_from_pdf(pdf_path)
        if not text.strip():
            raise ValueError("Could not extract text from PDF or PDF is empty")
        
        # Chunk text
        chunks = self.processor.chunk_text(text)
        if not chunks:
            raise ValueError("Could not create chunks from PDF text")
        
        # Store in vector database
        self.vector_store.add_documents(chunks, doc_name)
        
        return len(chunks)

# Main Agent
class WeatherRAGAgent:
    def __init__(self):
        print("Initializing Weather & Document Assistant...")
        self.nodes = GraphNodes()
        self.graph = self._build_graph()
        print("Agent initialized successfully!")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classify", self.nodes.classify_query_node)
        workflow.add_node("weather", self.nodes.weather_node)
        workflow.add_node("pdf_rag", self.nodes.pdf_rag_node)
        workflow.add_node("generate", self.nodes.generate_response_node)
        
        # Set entry point
        workflow.set_entry_point("classify")
        
        # Add conditional routing
        workflow.add_conditional_edges(
            "classify",
            self._route_query,
            {
                "weather": "weather",
                "pdf": "pdf_rag"
            }
        )
        
        # Add edges to generation
        workflow.add_edge("weather", "generate")
        workflow.add_edge("pdf_rag", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def _route_query(self, state: AgentState) -> str:
        """Route query based on classification"""
        return state["query_type"]
    @traceable(name="WeatherRAGAgent - process_query")
    def process_query(self, query: str) -> str:
        """Process a user query through the agent"""
        if not query.strip():
            return "Please provide a valid question."
        
        try:
            initial_state = AgentState(
                messages=[],
                query=query.strip(),
                query_type="",
                weather_data=None,
                pdf_context=None,
                final_response=None
            )
            
            result = self.graph.invoke(initial_state)
            return result.get("final_response", "I couldn't process your request. Please try again.")
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return "I encountered an error while processing your request. Please try again or rephrase your question."

# Environment check function
def check_environment():
    """Check if environment variables are properly set"""
    print("=== Environment Variable Check ===")
    
    azure_endpoint = os.getenv("AZURE_API_BASE")
    azure_key = os.getenv("AZURE_API_KEY")
    weather_key = os.getenv("OPENWEATHER_API_KEY")
    
    print(f"AZURE_API_BASE: {azure_endpoint}")
    print(f"AZURE_API_KEY: {'*' * 10 if azure_key else 'None'}")
    print(f"OPENWEATHER_API_KEY: {'*' * 10 if weather_key else 'None'}")
    
    missing = []
    if not azure_endpoint:
        missing.append("AZURE_API_BASE")
    if not azure_key:
        missing.append("AZURE_API_KEY")
    if not weather_key:
        missing.append("OPENWEATHER_API_KEY")
    
    if missing:
        print(f"\n❌ Missing environment variables: {', '.join(missing)}")
        print("The agent will work with limited functionality.")
    else:
        print("\n✅ All environment variables are set!")
    
    return len(missing) == 0

# Utility function for testing
def test_agent():
    """Simple test function"""
    try:
        print("\n=== Testing Weather & Document Assistant ===")
        
        # Check environment first
        env_ok = check_environment()
        
        agent = WeatherRAGAgent()
        
        # Test weather query
        print("\n--- Testing Weather Query ---")
        weather_response = agent.process_query("What's the weather in London?")
        print("Weather Response:", weather_response)
        
        # Test PDF query
        print("\n--- Testing PDF Query ---")
        pdf_response = agent.process_query("What are the main topics in the document?")
        print("PDF Response:", pdf_response)
        
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

