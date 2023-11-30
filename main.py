from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertModel, BertTokenizer
import torch
from code import get_contextual_embeddings, query_qdrant_for_context

csv_path = "/Users/sania/Downloads/bigBasketProducts.csv"
df = load_and_preprocess_data(csv_path)

app = FastAPI()

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

class Query(BaseModel):
    query: str

@app.post("/get_contextual_answer")
async def get_contextual_answer(query: Query):
    try:
        # Get contextual embeddings using the language model
        tokens = tokenizer(query.query, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        # Query Qdrant for similar embeddings (you can adapt this part based on your needs)
        similar_points = query_qdrant_for_context(query.query, qdrant)
        # For simplicity, returning the first result as an answer
        answer = similar_points['points'][0]['id']

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
