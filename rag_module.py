import requests
import json

def generate_recommendation(query, context):
    """
    Generates a recommendation using a local LLM via the Ollama API.

    Args:
        query (str): The flattened API specification text to analyze.
        context (str): The security guidelines retrieved from the knowledge base.

    Returns:
        str: An actionable, concise recommendation from the LLM, or an error message.
    """
    prompt = f"""
    You are a security analyst reviewing an OpenAPI specification. Your task is to identify potential security vulnerabilities based on the provided API details and known security guidelines. For each vulnerability you find, provide a clear, actionable recommendation to fix it.

    Security Guidelines:
    {context}

    API Specification Details:
    {query}

    Please provide a detailed report of potential vulnerabilities and recommendations.
    """

    ollama_api_url = "http://localhost:11434/api/generate"

    payload = {
        "model": "gemma2:2b",
        "prompt": prompt,
        "stream": False
    }

    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(ollama_api_url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()

        result = response.json()
        return result['response'].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling local LLM: {e}")
        return "Could not generate a recommendation. Please ensure the Ollama service is running and the 'gemma2:2b' model is installed."
