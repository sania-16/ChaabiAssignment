# Chaabi Assignment

Given a data source (csv per say), build a Query Engine for English language

Deliverables:
- Implement vector embeddings on the given dataset and store them in a Vector DB like Qdrant.
- Implement an LLM on the DB that can give contextual answers to the queries strictly from the database.
- Wrap this LLM as an API using any framework.

Instructions:
- Save the code.py and main.py files in the same directory.
- Replace "/Users/sania/Downloads/bigBasketProducts.csv" with the actual path to your CSV file in main.py and code.py.
- Install required libraries using pip install pandas gensim qdrant-client transformers fastapi uvicorn.
- Run the FastAPI application with
    uvicorn main:app --reload.
- Access the API documentation at http://127.0.0.1:8000/docs to test your API interactively.

Sample Queries:

1. curl -X 'POST' \
   'http://127.0.0.1:8000/get_contextual_answer' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "query": "Find similar products to XYZ"
  }'

2. curl -X 'POST' \
  'http://127.0.0.1:8000/get_contextual_answer' \
   -H 'accept: application/json' \
   -H 'Content-Type: application/json' \
   -d '{
   "query": "Find products with rating greater than 4."
  }'


