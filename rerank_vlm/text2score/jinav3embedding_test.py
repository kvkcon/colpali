from sentence_transformers import SentenceTransformer, CrossEncoder


model = SentenceTransformer(
    "/data/hf_models/jina-embeddings-v3",
    trust_remote_code=True,
    # cache_dir="/data/hf_models/"
)


query = "What is the weather like in Berlin today?"
query_embedding = model.encode(query, task='retrieval.query')


documents = ["Berlin is a city with variable weather.", "Today in Berlin, it is sunny.", "Weather in Berlin can be unpredictable."]
document_embeddings = model.encode(documents, task='retrieval.passage')


similarity_scores = []
for doc in document_embeddings:
    similarity = float(((query_embedding * doc).sum()) / (query_embedding.norm() * doc.norm()))
    similarity_scores.append(similarity)


top_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:25]


top_documents = [documents[i] for i in top_indices]
for doc in top_documents:
    print(doc)