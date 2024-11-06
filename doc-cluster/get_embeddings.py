from sentence_transformers import SentenceTransformer

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# List of documents
documents = [
    "Document 1 text goes here.",
    "Document 2 text goes here.",
    # Add more documents as needed
]

# Compute embeddings
embeddings = model.encode(documents)

# 'embeddings' is a list of numpy arrays, each corresponding to a document

