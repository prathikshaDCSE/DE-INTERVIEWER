"""
settings.py

Central configuration file for the DE-INTERVIEWER project.

This file loads all environment variables from the .env file
and makes them available to the entire application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class Settings:
    """
    Application Configuration
    """

    # ==========================================================
    # Application Information
    # ==========================================================

    APP_NAME = "DE-INTERVIEWER"

    APP_VERSION = "1.0.0"

    DEBUG = True

    # ==========================================================
    # Gemini Configuration
    # ==========================================================

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    GEMINI_MODEL = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.6-flash"
    )

    # ==========================================================
    # Google Cloud Configuration
    # ==========================================================

    GCP_PROJECT_ID = os.getenv(
        "GCP_PROJECT_ID",
        "de-interviewer"
    )

    BIGQUERY_DATASET = os.getenv(
        "BIGQUERY_DATASET",
        "de_Interviewer"
    )

    SERVICE_ACCOUNT_FILE = os.getenv(
    "SERVICE_ACCOUNT_FILE",
    "secrets\service_account.json"
)

    # ==========================================================
    # OAuth Configuration
    # ==========================================================

    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

    # ==========================================================
    # Streamlit Configuration
    # ==========================================================

    STREAMLIT_PORT = int(
        os.getenv("STREAMLIT_PORT", 8501)
    )


# Create a single settings object
settings = Settings()