from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Reply with exactly: Gemini is connected successfully!"
)

print(response.text)