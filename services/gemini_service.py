"""
gemini_service.py

This module handles all communication with the Gemini API.
"""

from google import genai
from config.settings import settings


class GeminiService:
    """
    Service class for interacting with Gemini.
    """

    def __init__(self):
        self.client = genai.Client(
            api_key=settings.GOOGLE_API_KEY
        )

    def generate_response(self, prompt: str):
        """
        Send a prompt to Gemini and return the response.
        """

        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt
        )

        return response.text


# Create one reusable service object
gemini_service = GeminiService()