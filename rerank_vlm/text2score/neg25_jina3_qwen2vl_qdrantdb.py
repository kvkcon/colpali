import torch
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset, load_from_disk, Features, Value, Sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

def main():
    # Load the model and tokenizer
    model_name = "/data/hf_models/jina-embeddings-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        ).to('cuda')
    model.eval()

    # Load dataset
    dataset_path = "/data/hf_datasets/ocr_erank_train_set_colpali_qwen2vl_train118659_vllm_4096/part_1"
    dataset = load_from_disk(dataset_path)
    
    # Initialize Qdrant
    qdrant_client = QdrantClient(":memory:")
    collection_name = "embeddings"
    vector_size = model.config.hidden_size

    # Create collection
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )

    # Process text embeddings in batches
    batch_size = 32
    
    def compute_embeddings(texts):
        with torch.no_grad():
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to('cuda')
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].to(torch.float32).cpu().numpy()
            # embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            return embeddings

    # Index all text embeddings
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_texts = dataset[i:i+batch_size]['text']
        embeddings = compute_embeddings(batch_texts)
        
        points = [
            models.PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={"original_index": idx}
            )
            for idx, embedding in enumerate(embeddings, start=i)
        ]
        
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )

    # Process queries and find similar texts
    new_hf_dataset_dict = {
        "query": [],
        "positive_index": [],
        "topk_indices": [],
        "negative_indices": [],
        "scores": []
    }

    for i in tqdm(range(0, len(dataset), batch_size),desc="Processing queries..."):
        batch_queries = dataset[i:i+batch_size]['query']
        query_embeddings = compute_embeddings(batch_queries)

        for ki, query_embedding in enumerate(query_embeddings):
            positive_index = i + ki
            
            search_result = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=26
            )
            
            scores_k = [r.score for r in search_result]
            top_k_indices = [r.payload["original_index"] for r in search_result]
            negative_indices = [idx for idx in top_k_indices if idx != positive_index]
            
            new_hf_dataset_dict["query"].append(dataset[positive_index]["query"])
            new_hf_dataset_dict["positive_index"].append(positive_index)
            new_hf_dataset_dict["topk_indices"].append(top_k_indices)
            new_hf_dataset_dict["negative_indices"].append(negative_indices)
            new_hf_dataset_dict["scores"].append(scores_k)

    # Create and save new dataset
    features = Features({
        "query": Value("string"),
        "positive_index": Value("int32"),
        "topk_indices": Sequence(Value("int32")),
        "negative_indices": Sequence(Value("int32")),
        "scores": Sequence(Value("float32"))
    })

    new_dataset = Dataset.from_dict(new_hf_dataset_dict, features=features)
    output_path = "/data/hf_datasets/ocr_t2s_train_set_colpali_jina3_train5000_4096/"
    new_dataset.save_to_disk(output_path)

    print(f"Dataset saved to {output_path}")



    # validate
    loaded_dataset = load_from_disk(output_path)
    print(f"Loaded dataset sample: {loaded_dataset[0]}")
    print(f"Number of samples loaded: {len(loaded_dataset)}")

    print("New dataset has been created and saved.")

    #print random 6 samples
    print("Random 6 samples:")
    import random
    for i in range(6):
        random.seed()
        seed = random.randint(0, len(loaded_dataset)-1)
        print(f"Positive Index: {loaded_dataset[seed]['positive_index']}")
        print(f"top_k Indices: {loaded_dataset[seed]['topk_indices']}")#negative_indices
        print(f"Scores: {loaded_dataset[seed]['scores']}")
        print("\n")

if __name__ == "__main__":
    main()