import pprint
import random
import os
import json
from typing import List, cast
import argparse

import torch
from datasets import Dataset, load_dataset, Features, Value, Sequence, Image, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image as PILImage
import yaml
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
import stamina


# Set environment variable
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'



def main():
    ################################################################################################
    #                            118659 -> 59370(pdf+tatdqa) + 59289                               #
    #                                -  record ID in raw dataset                                   #
    #                                - filter 500 items                                            #
    #                                    - 300 items in the same source                            #
    #                                    - 200 items in others                                     #
    ################################################################################################

    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--data-source', type=int, choices=[0, 1], help="source pdf&tatdqa that accepts 0 or 1", required=True)
    args = parser.parse_args()

    data_source = args.data_source
    print(f"data source value: {data_source}")
    # Load configuration
    with open('config/data_gen/neg25_raw_config.yaml', 'r') as f:
        config = yaml.safe_load(f)


    ################################################################################################
    #                                                                                              #
    #                                index the dataset using Qdrant                                #
    #                                                                                              #
    ################################################################################################
    # print("embeddings_doc.shape", embeddings_doc.shape)
    vector_size = 128# embeddings_doc.shape[2]
    # print("vector_size", vector_size)
    # Creating a Qdrant client
    # global qdrant_client
    qdrant_client = QdrantClient(
        ":memory:",
        # path=config['db']['db_path'],
    )  # Use ":memory:" for in-memory database or "path/to/db" for persistent storage

    # Create a collection in Qdrant with a multivector configuration
    collection_name = config["db"]["collection_name"]
    qdrant_client.recreate_collection(
        collection_name=collection_name,  # the name of the collection
        on_disk_payload=True,  # store the payload on disk
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=100
        ),  # it can be useful to swith this off when doing a bulk upload and then manually trigger the indexing once the upload is done
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    always_ram=True,
                ),
            ),
            # quantization_config=models.ScalarQuantization(
            #     scalar=models.ScalarQuantizationConfig(
            #         type=models.ScalarType.INT8, # Scalar quantization allows you to reduce the number of bits used to 8
            #         quantile=0.99,
            #         always_ram=True,
            #     ),
            # ),
        ),
        # shard_number=2,
    )

    ################################################################################################
    #                                                                                              #
    #                                       load model ckpt                                        #
    #                                                                                              #
    ################################################################################################    
    device = torch.device(config['device'])
    print(f"Device used: {device}")

    model_name = config['model']['name']

    # Load model
    # colpali_model = ColPali.from_pretrained(
    #     model_name, 
    #     cache_dir=config['model']['cache_dir'],
    #     torch_dtype=torch.bfloat16, 
    #     device_map=device
    # )
    # processor = ColPaliProcessor.from_pretrained(model_name)
    # colpali_model = colpali_model.eval()

    # if not isinstance(processor, BaseVisualRetrieverProcessor):
    #     raise ValueError("Processor should be a BaseVisualRetrieverProcessor")

    # # Load the entire dataset
    # raw_dataset = cast(Dataset, load_dataset(config['dataset']['input_path'], split=config['dataset']['split']))

    # if data_source == 0:
    #     # handle dataset from pdf&tatdqa source
    #     dataset = raw_dataset.filter(lambda x: x['source'] in config['dataset']['source'])
    #     print("data from source pdf & tatdqa Handling")
    # elif data_source == 1:
    #     # handle dataset not from pdf&tatdqa source
    #     dataset = raw_dataset.filter(lambda x: x['source'] not in config['dataset']['source'])
    #     print("data from source docvqa & arxiv_qa & Infographic-VQA Handling")
    # else:
    #     print("Full data source Handling")
    # print("Dataset loaded: ", raw_dataset)
    # print("Dataset filtered to be handle", dataset)
    dataset = [0 for _ in range(118659)]
    print("Dataset length: ", len(dataset))

    # Process all images
    print("Processing images...")


    # upload the vectors to qdrant
    @stamina.retry(on=Exception, attempts=3)
    def upsert_to_qdrant(batch):
        try:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=False,
            )
        except Exception as e:
            print(f"Error during upsert: {e}")
            return False
        return True

    batch_size = config['batch_size']['image']
    image_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: (x),
    )
    # Use tqdm to create a progress bar
    for batch_idx, batch_doc in enumerate(tqdm(image_dataloader, desc="Indexing Progress")):
        
        # embeddings_doc equals to the random vec of 2*128
        embeddings_doc = torch.randn(batch_size, 2, 128)
        # print(f"embeddings_doc: {embeddings_doc}")

        # Prepare points for Qdrant
        points = []
        for j, embedding in enumerate(embeddings_doc):
            # Convert the embedding to a list of vectors
            multivector = embedding[0].cpu().float().numpy().tolist()
            # print(f"multivector: {multivector}")
            points.append(
                models.PointStruct(
                    id=batch_idx*batch_size+j,  # we just use the index as the ID
                    vector=multivector,  # This is now a list of vectors
                )
            )
        # Upload points to Qdrant
        try:
            upsert_to_qdrant(points)
        # clown level error handling here
        except Exception as e:
            print(f"Error during upsert: {e}")
            continue

    print("Indexing complete!")
    qdrant_client.update_collection(
        collection_name=collection_name,
        hnsw_config=models.HnswConfigDiff(
            m=64,  # Increase the number of edges per node from the default 16 to 32
            ef_construct=64,  # Increase the number of neighbours from the default 100 to 200
        )
    )
    while True:
        collection_info = qdrant_client.get_collection(collection_name=collection_name)
        if collection_info.status == models.CollectionStatus.GREEN:
            # Collection status is green, which means the indexing is finished
            break
    print("HNSW status is green, which means the indexing is finished")


    # # Example usage
    # query_text = "declassified data"
    # results = search_images_by_text(query_text)

    # for result in results.points:
    #     print(result)

    # Process all queries
    print("Processing queries...")
    query_dataloader = DataLoader(
        dataset=dataset,
        batch_size=config['batch_size']['query'],
        shuffle=False,
        collate_fn=lambda x: (x),
    )

    # query_embeddings: List[torch.Tensor] = []
    top_k = 26
    new_hf_dataset_dict = {
        "positive_index": [],
        "topk_indices": [],
        "scores": []
    }
    for batch_idx, batch_query in enumerate(tqdm(query_dataloader, desc="Calculating scores and selectin topK negative samples...")):

        embeddings_query = torch.randn(config['batch_size']['query'], 2, 128)
        for ki in range(len(embeddings_query)):
            multivector_query = embeddings_query[ki][0].cpu().float().numpy().tolist()
            # query_embeddings.extend(list(torch.unbind(embeddings_query.to("cpu"))))
            # Calculate scores and select top 26 negative samples for each query batch
            search_result = qdrant_client.query_points(
                # collection_name=collection_name, 
                # query=multivector_query, 
                # limit=top_k,
                collection_name=collection_name, 
                query=multivector_query,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=2.0,
                    )
                ),
                limit=top_k
            )
            positive_index = batch_idx*config['batch_size']['query']+ki
            scores_k = [r.score for r in search_result.points]
            top_k_indices = [r.id for r in search_result.points] 

            # Directly append to the dictionary
            new_hf_dataset_dict["positive_index"].append(positive_index)
            new_hf_dataset_dict["topk_indices"].append(top_k_indices)
            new_hf_dataset_dict["scores"].append(scores_k)

    # Save the new dataset
    print("Saving the new dataset...")

    features = Features({
        "positive_index": Value("int32"),
        "topk_indices": Sequence(Value("int32")),
        "scores": Sequence(Value("float32"))
    })
    hf_dataset = Dataset.from_dict(new_hf_dataset_dict, features=features)

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
        seed = random.randint(0, len(loaded_dataset)-1)
        print(f"Positive Index: {loaded_dataset[seed]['positive_index']}")
        print(f"top_k Indices: {loaded_dataset[seed]['topk_indices']}")#negative_indices
        print(f"Scores: {loaded_dataset[seed]['scores']}")
        print("\n")


if __name__ == "__main__":  
    main()