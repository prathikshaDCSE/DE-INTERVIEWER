from database.bigquery import bigquery_service

tables = bigquery_service.list_tables()

print(tables)