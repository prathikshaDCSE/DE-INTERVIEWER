from services.gemini_service import gemini_service

response = gemini_service.generate_response(
    "Reply only with: DE-INTERVIEWER Service Working!"
)

print(response)