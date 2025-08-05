# 🌦️ RAG Weather & Web Search Assistant

An intelligent assistant powered by LangChain, LangGraph, and Streamlit that performs two key tasks:
- Provides **real-time weather data** using OpenWeatherMap.
- Answers **questions from uploaded PDF documents** using **Retrieval-Augmented Generation (RAG)**.
- Searches query online.

Built with ❤️ using Azure OpenAI, ChromaDB, and a modular multi-tool agent system.

---

## ✅ Features

- 🌤️ Real-time weather data retrieval via OpenWeatherMap API  
- 📄 Ask questions from any uploaded PDF using RAG (LangChain + ChromaDB)  
- 🔍 Web Search available
- 🧠 Modular LangGraph Agent with tool calling  
- 💻 Streamlit UI for easy interaction

---

## ✅ Run Locally

### 1. Clone the Repository

    git clone https://github.com/vidushigoel31/RAG-Weather-WebSearch.git
    cd RAG-Weather-WebSearch

### 2. Create and Activate Virtual Environment

    # For Windows:
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux:
    python3 -m venv venv
    source venv/bin/activate

### 3. Install Requirements

    pip install -r requirements.txt

### 4. Set Environment Variables

Create a `.env` file in the root directory and add the following keys:

    AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
    AZURE_OPENAI_API_VERSION=your_azure_openai_version
    AZURE_OPENAI_API_KEY=your_azure_openai_key
    AZURE_OPENAI_ENDPOINT=your_azure_endpoint
    OPENWEATHER_API_KEY=your_openweather_key

> ⚠️ Do NOT share or commit your `.env` file.

### 5. Run the Streamlit App

    streamlit run app.py

---

## ✅ Project Structure

    RAG-Weather-WebSearch/
    ├── app.py                      # Streamlit app
    ├── agent_system.py             # LangGraph agent and RAG logic
    ├── web_search.py               # Web search fallback tool
    ├── requirements.txt            # Python dependencies
    ├── tests/
    │   └── agent_test.py           # Unit tests for the agent logic
    └── README.md                   # Project documentation

---

## ✅ Tech Stack

- LangChain  
- LangGraph  
- Azure OpenAI  
- OpenWeatherMap API  
- ChromaDB  
- Streamlit
- duckduckgo-search 


---

## ✅ License

This project is licensed under the MIT License. See `LICENSE` file for more details.

---

## ✅ Contributing

Feel free to fork this repo, raise issues, or submit pull requests. Contributions are welcome!

