from google.cloud import bigquery

client = bigquery.Client()

query = """
SELECT 
COUNT(DISTINCT id_adv) as counter
FROM `movilidadcovid.Movilidad.Hermosillo`
WHERE DATE(timestamp) BETWEEN "2020-01-01"
AND "2020-12-31"
"""

results = client.query(query)

for row in results:
    print(row['counter'])
