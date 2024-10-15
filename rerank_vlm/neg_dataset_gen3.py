import pprint
import random
import os
import json
from typing import List, cast

import torch
from datasets import Dataset, load_dataset, Features, Value, Sequence, Image, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image as PILImage
import yaml

from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

# Set environment variable
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Load configuration
with open('config/data_gen/neg25_raw_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device(config['device'])
print(f"Device used: {device}")

model_name = config['model']['name']

# Load model
model = ColPali.from_pretrained(
    model_name, 
    cache_dir=config['model']['cache_dir'],
    torch_dtype=torch.bfloat16, 
    device_map=device
)
processor = ColPaliProcessor.from_pretrained(model_name)
model = model.eval()

if not isinstance(processor, BaseVisualRetrieverProcessor):
    raise ValueError("Processor should be a BaseVisualRetrieverProcessor")

# Load the entire dataset
dataset = cast(Dataset, load_dataset(config['dataset']['input_path'], split="train"))
print("Dataset loaded: ", dataset)

# Process all images
print("Processing images...")
image_dataloader = DataLoader(
    dataset=dataset["image"],
    batch_size=config['batch_size']['image'],
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
    batch_size=config['batch_size']['query'],
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
    
    positive_index = i
    top_indices = scores.argsort()[::-1]
    top_k_indices = top_indices[:26]
    negative_indices = [idx for idx in top_k_indices if idx != positive_index][:25]
    scores_k = scores[negative_indices]
    
    new_entry = {
        "query": dataset[i]["query"],
        "positive_image": dataset[i]["image"],
        "positive_index": positive_index,
        "topk_indices": top_k_indices,
        "negative_indices": negative_indices,
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

hf_dataset = Dataset.from_dict(hf_dataset_dict)

features = Features({
    "query": Value("string"),
    "positive_image": Image(),
    "positive_index": Value("int32"),
    "top_k_indices": Sequence(Value("int32")),
    "negative_indices": Sequence(Value("int32")),
    "scores": Sequence(Value("float32"))
})

hf_dataset = hf_dataset.cast(features)

output_dir = config['dataset']['output_dir']
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