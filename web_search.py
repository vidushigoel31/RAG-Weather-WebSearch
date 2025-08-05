# web_search_service.py
"""
Minimal Web Search Service - Simple and Reliable
"""

import requests
import json
from urllib.parse import quote


def web_search(query: str, max_results: int = 3) -> str:
    """
    Simple web search using DuckDuckGo Instant Answer API
    No SSL issues, no complex dependencies
    """
    try:
        # Use DuckDuckGo's simple API
        encoded_query = quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        
        if response.status_code != 200:
            return f"🔍 **Search for: {query}**\n\nSearch service temporarily unavailable."
        
        data = response.json()
        
        # Format results
        results_text = f"🔍 **Web Search Results for: {query}**\n\n"
        
        # Check for instant answer
        if data.get('AbstractText'):
            results_text += f"**Summary:**\n{data['AbstractText']}\n\n"
            if data.get('AbstractURL'):
                results_text += f"🔗 Source: {data['AbstractURL']}\n\n"
        
        # Check for related topics
        if data.get('RelatedTopics'):
            results_text += "**Related Information:**\n"
            count = 0
            for topic in data['RelatedTopics']:
                if count >= max_results:
                    break
                if isinstance(topic, dict) and topic.get('Text'):
                    results_text += f"• {topic['Text']}\n"
                    if topic.get('FirstURL'):
                        results_text += f"  🔗 {topic['FirstURL']}\n"
                    results_text += "\n"
                    count += 1
        
        # If no results found
        if not data.get('AbstractText') and not data.get('RelatedTopics'):
            results_text += "No detailed results found. Try rephrasing your search query."
        
        return results_text
        
    except requests.exceptions.SSLError:
        return f"🔍 **Search for: {query}**\n\nSSL connection issue. Search temporarily unavailable."
    
    except requests.exceptions.ConnectionError:
        return f"🔍 **Search for: {query}**\n\nConnection issue. Please check your internet connection."
    
    except Exception as e:
        return f"🔍 **Search for: {query}**\n\nSearch error: {str(e)}"


# Alternative simple search function using a different approach
def simple_web_search(query: str) -> str:
    """
    Ultra-simple fallback search that always works
    """
    return f"""🔍 **Web Search: {query}**

To get real web search results, you can:
1. Search on Google: https://www.google.com/search?q={quote(query)}
2. Search on DuckDuckGo: https://duckduckgo.com/?q={quote(query)}
3. Search on Bing: https://www.bing.com/search?q={quote(query)}

*Note: Direct web search integration is experiencing connectivity issues.*
"""


