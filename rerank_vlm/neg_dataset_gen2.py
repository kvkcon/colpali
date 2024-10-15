import pprint
import random
from typing import List, cast

import torch
from datasets import Dataset, load_dataset, Features, Value, Sequence, Image, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image as PILImage

from colpali_engine.models import ColPali,ColQwen2, ColQwen2Processor, ColPaliProcessor
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device

import os
import json

def get_torch_device(device_str):
    # Implement this function based on your needs
    return torch.device(device_str if torch.cuda.is_available() else "cpu")

device = get_torch_device("cuda")
print(f"Device used: {device}")

# model_name = "vidore/colqwen2-v0.1"
model_name = "vidore/colpali-v1.2"

# Load model
model = ColPali.from_pretrained(
    model_name, 
    cache_dir='/data/hf_models/',
    torch_dtype=torch.bfloat16, 
    device_map=device#"auto"
)
processor = ColPaliProcessor.from_pretrained(model_name)
model = model.eval()

if not isinstance(processor, BaseVisualRetrieverProcessor):
    raise ValueError("Processor should be a BaseVisualRetrieverProcessor")

# Load the entire dataset
dataset = cast(Dataset, load_dataset("/data/hf_datasets/colpali_train_set/", split="train"))#[:50]
print("Dataset loaded: ",dataset)

# Process all images
print("Processing images...")
image_dataloader = DataLoader(
    dataset=dataset["image"],
    batch_size=4,
    shuffle=False,
    collate_fn=lambda x: processor.process_images(x),
)

image_embeddings: List[torch.Tensor] = []
for batch_doc in tqdm(image_dataloader):
    with torch.no_grad():
        batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
        embeddings_doc = model(**batch_doc)
    image_embeddings.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

# Process all queries
print("Processing queries...")
query_dataloader = DataLoader(
    dataset=dataset["query"],
    batch_size=32,
    shuffle=False,
    collate_fn=lambda x: processor.process_queries(x),
)

query_embeddings: List[torch.Tensor] = []
for batch_query in tqdm(query_dataloader):
    with torch.no_grad():
        batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
        embeddings_query = model(**batch_query)
    query_embeddings.extend(list(torch.unbind(embeddings_query.to("cpu"))))

# Calculate scores and select top 26 negative samples for each query
print("Calculating scores and selecting negative samples...")
new_dataset = []

for i, query_embedding in enumerate(tqdm(query_embeddings)):
    scores = processor.score([query_embedding], image_embeddings).cpu().numpy()[0]
    
    # Get the index of the positive sample
    positive_index = i
    
    # Get indices of top 27 scores (excluding the positive sample)
    top_indices = scores.argsort()[::-1]
    # negative_indices = [idx for idx in top_indices if idx != positive_index][:26]
    top_k_indices = top_indices[:26]
    negative_indices = [idx for idx in top_k_indices if idx != positive_index][:25]
    scores_k = scores[negative_indices]
    
    # Create a new entry for the dataset
    new_entry = {
        "query": dataset[i]["query"],
        "positive_image": dataset[i]["image"],
        "positive_index": positive_index,
        "topk_indices": top_k_indices,
        "negative_indices": negative_indices,
        # "negative_images": [dataset[int(idx)]["image"] for idx in negative_indices],
        "scores": scores_k
    }
    new_dataset.append(new_entry)

hf_dataset_dict = {
    "query": [],
    "positive_image": [],
    "positive_index": [],
    "top_k_indices": [],
    "negative_indices": [],
    "scores": []
}
for entry in tqdm(new_dataset, desc="Converting dataset"):
    hf_dataset_dict["query"].append(entry["query"])
    hf_dataset_dict["positive_image"].append((entry["positive_image"]))
    hf_dataset_dict["positive_index"].append(entry["positive_index"])
    hf_dataset_dict["top_k_indices"].append(entry["topk_indices"])
    hf_dataset_dict["negative_indices"].append(entry["negative_indices"])
    hf_dataset_dict["scores"].append(entry["scores"])

# Save the new dataset
print("Saving the new dataset...")

# create new dataset
hf_dataset = Dataset.from_dict(hf_dataset_dict)

# define features
features = Features({
    "query": Value("string"),
    "positive_image": Image(),
    "positive_index": Value("int32"),
    "top_k_indices": Sequence(Value("int32")),
    "negative_indices": Sequence(Value("int32")),
    "scores": Sequence(Value("float32"))
})

hf_dataset = hf_dataset.cast(features)

# save
output_dir = "/data/hf_datasets/rerank_train_set_colpali_1_2/"
os.makedirs(output_dir, exist_ok=True)
hf_dataset.save_to_disk(output_dir)

print(f"Dataset saved to {output_dir}")

# validate
loaded_dataset = load_from_disk(output_dir)
print(f"Loaded dataset sample: {loaded_dataset[0]}")
print(f"Number of samples loaded: {len(loaded_dataset)}")

print("New dataset has been created and saved.")

#print random 6 samples
print("Random 6 samples:")

for i in range(6):
    random.seed()
    seed = random.randint(0, len(new_dataset))
    print(f"Query: {new_dataset[seed]['query']}")
    print(f"Positive Image: {new_dataset[seed]['positive_image']}")
    print(f"Positive Index: {new_dataset[seed]['positive_index']}")
    print(f"top_k Indices: {new_dataset[seed]['top_k_indices']}")
    print(f"Scores: {new_dataset[seed]['scores']}")
    print("\n")