"""
bigquery.py

Handles all communication with Google BigQuery.
"""

from google.cloud import bigquery
from google.oauth2 import service_account

from config.settings import settings


class BigQueryService:
    """
    Service class for interacting with Google BigQuery.
    """

    def __init__(self):

        # Load credentials from the service account JSON
        credentials = service_account.Credentials.from_service_account_file(
            settings.SERVICE_ACCOUNT_FILE
        )

        # Create BigQuery client
        self.client = bigquery.Client(
            project=settings.GCP_PROJECT_ID,
            credentials=credentials
        )

        self.dataset = settings.BIGQUERY_DATASET

    def list_tables(self):
        """
        Returns all tables inside the dataset.
        """

        dataset_ref = self.client.dataset(self.dataset)

        tables = self.client.list_tables(dataset_ref)

        return [table.table_id for table in tables]


# Create a reusable instance
bigquery_service = BigQueryService()