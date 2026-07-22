from config.settings import settings

print("Application :", settings.APP_NAME)
print("Version     :", settings.APP_VERSION)
print("Gemini Model:", settings.GEMINI_MODEL)
print("Project ID  :", settings.GCP_PROJECT_ID)
print("Dataset     :", settings.BIGQUERY_DATASET)
print("API Key     :", settings.GOOGLE_API_KEY[:10] + "...")