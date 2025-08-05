import pytest
from agent_system import WeatherService, WeatherData
import responses
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
@responses.activate
def test_weather_service_success():
    city = "London"
    mock_response = {
        "name": city,
        "main": {"temp": 22, "humidity": 70},
        "weather": [{"description": "sunny"}],
        "wind": {"speed": 5.5}
    }
    responses.add(
        responses.GET,
        "http://api.openweathermap.org/data/2.5/weather",
        json=mock_response,
        status=200
    )

    service = WeatherService()
    data = service.get_weather(city)

    assert isinstance(data, WeatherData)
    assert data.city == city
    assert data.temperature == 22
    assert data.humidity == 70
from agent_system import LLMService

def test_llm_classification_fallback_weather():
    service = LLMService()
    query = "Tell me the weather in Mumbai"
    result = service.classify_query(query)
    assert result == "weather"

def test_llm_classification_fallback_pdf():
    service = LLMService()
    query = "Summarize the document"
    result = service.classify_query(query)
    assert result == "pdf"

from agent_system import PDFProcessor, VectorStore

def test_chunk_text():
    processor = PDFProcessor()
    long_text = "Hello world. " * 200  # long enough for multiple chunks
    chunks = processor.chunk_text(long_text)
    assert len(chunks) > 1
    assert all(isinstance(chunk, str) for chunk in chunks)

def test_vector_store_add_and_search(tmp_path):
    vs = VectorStore()
    texts = ["LangChain is cool.", "OpenAI powers GPT-4.", "Weather today is sunny."]
    vs.add_documents(texts, doc_name="test_doc")
    
    results = vs.search_similar("What powers GPT?")
    assert any("OpenAI" in doc for doc in results)
