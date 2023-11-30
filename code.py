import pandas as pd
from gensim.models import Word2Vec
from qdrant_client import QdrantClient
from transformers import BertModel, BertTokenizer
import torch

dataset_path = 'bigBasketProducts.csv'
df = pd.read_csv(dataset_path)

# Get information about the dataset
df.info()

# Drop rows with missing values in the 'product_name' column
df = df.dropna(subset=['product'])

df.to_csv('cleaned_dataset.csv', index=False)

import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK resources (only needed once)
nltk.download('punkt')

dataset_path = 'cleaned_dataset.csv'
df = pd.read_csv(dataset_path)

#Tokenization is the process of splitting text into individual words
tokenized_names = [word_tokenize(name.lower()) for name in df['product']]

#Train a Word2Vec model on the tokenized product names
model = Word2Vec(sentences=tokenized_names, vector_size=150, window=5, min_count=1, workers=4)

# Save the model
model.save('word2vec_model.model')

# Load the model
loaded_model = Word2Vec.load('word2vec_model.model')

from qdrant_client import QdrantClient
qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance, for testing, CI/CD
# OR
#client = QdrantClient(path="path/to/db")  # Persists changes to disk, fast prototyping

qdrant_endpoint = "http://localhost:6333"
qdrant = QdrantClient(qdrant_endpoint) # Connect to existing Qdrant instance, for production


# Example: Iterate through your dataset and store embeddings in Qdrant
for index, row in df.iterrows():
    product_id = str(row['index'])  # Assuming 'rating' is a column in our dataset
    product_name = row['product']

    # Retrieve the vector for each tokenized word in the product name
    vectors = [loaded_model.wv[token.lower()] for token in word_tokenize(product_name.lower())]

    # Combine the vectors (for example, by taking the mean)
    vector = [sum(component) / len(component) for component in zip(*vectors)]

    # Store the vector in Qdrant
    qdrant.upsert(collection_name='products', points=[{"id": product_id, "vector": vector}])


from transformers import BertModel, BertTokenizer
import torch
from qdrant_client import QdrantClient

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Connect to Qdrant
qdrant_endpoint = "http://localhost:6333"
qdrant = QdrantClient(qdrant_endpoint)

def get_contextual_embeddings(query):
    # Tokenize query
    tokens = tokenizer(query, return_tensors='pt')

    # Forward pass through the model to get contextual embeddings
    with torch.no_grad():
        outputs = model(**tokens)

    # Extract contextual embeddings from the model output
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return embeddings

def query_qdrant_for_context(query, qdrant_client):
    # Get contextual embeddings using the language model
    contextual_embeddings = get_contextual_embeddings(query)

    # Query Qdrant for similar embeddings
    similar_points = qdrant_client.search(
        collection_name='products',
        query_vector=contextual_embeddings.tolist(),
        vector_name='vector'  # Assuming 'vector' is the name of the vector field in Qdrant
    )

    return similar_points
