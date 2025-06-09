from typing import Optional, List
import os
import time

# Google Gemini API
from google import genai
from google.oauth2 import service_account
from google.genai.types import GenerateContentConfig


from dotenv import load_dotenv

class LLMClient:
    def __init__(self):
        pass

    def call_llm(self):
        pass
    
class GeminiClient(LLMClient):
    
    def __init__(self):
        """Initialize the Gemini API client with GCP credentials."""
        load_dotenv(override=True)
        creds_path = os.getenv("GCP_CREDENTIALS")
        scopes = [
            "https://www.googleapis.com/auth/generative-language",
            "https://www.googleapis.com/auth/cloud-platform",
        ]
        credentials = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=scopes
        )
        self.client = genai.Client(
            vertexai=True,
            project=os.getenv("GCP_PROJECT"),
            location=os.getenv("GCP_REGION"),
            credentials=credentials
        )
        
    def call_llm(self,
                 prompt: str,
                 model: str,
                 batch_size: int,
                 temperature: float,
                 output_format: str = "text",      # "text" or "json"
                 json_schema: dict = None,
                 max_candidates: int = 8
                 ) -> List[str]:
        """
        output_format: "text" for plain text, "json" for structured JSON.
        json_schema: a JSON Schema dict if output_format == "json".
        """
        results = []
        for i in range(0, batch_size, max_candidates):
            current = min(max_candidates, batch_size - i)
            config = GenerateContentConfig(
                temperature=temperature,
                candidate_count=current,
                # Set MIME type based on desired output
                response_mime_type=(
                    "application/json"
                    if output_format == "json"
                    else "text/plain"
                ),
                # Only include schema when requesting JSON
                response_schema=json_schema if output_format == "json" else None
            )
            resp = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
            # Extract text (for JSON you can json.loads on resp.text)
            results.extend([c.content.parts[0].text for c in resp.candidates])
        return results