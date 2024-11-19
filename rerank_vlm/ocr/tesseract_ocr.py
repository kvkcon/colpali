import pytesseract
from PIL import Image
from typing import List, cast
import yaml
from datasets import Dataset, load_dataset, Features, Value, Sequence, Image, load_from_disk
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import gc

with open('config/data_gen/neg25_raw_config_test.yaml', 'r') as f:
    config = yaml.safe_load(f)

dataset = cast(Dataset, load_dataset(config['dataset']['input_path'], split=config['dataset']['split']))
print("Dataset loaded: ", dataset)

def process_image(image):
    text = pytesseract.image_to_string(image)
    return text

new_data = {
    'query': [],
    'text': [],
    'raw_id': []
}

progress_bar = tqdm(total=len(dataset), desc='Processing images')

def process_and_update(idx, row):
    try:
        text = process_image(row['image'])
        new_data['query'].append(row['query'])
        new_data['text'].append(text)
        new_data['raw_id'].append(idx)
    except Exception as exc:
        print(f'Image {idx} generated an exception: {exc}')
    finally:
        progress_bar.update(1)

def save_data(new_data, output_dir, part):
    part_output_dir = os.path.join(output_dir, f'part_{part}')
    os.makedirs(part_output_dir, exist_ok=True)
    part_dataset = Dataset.from_dict(new_data)
    part_dataset.save_to_disk(part_output_dir)
    print(f"Part {part} saved to {part_output_dir}")
    new_data.clear()
    gc.collect() 

output_dir = config['dataset']['output_dir']
os.makedirs(output_dir, exist_ok=True)

part = 1
count = 0
batch_size = 500

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_and_update, idx, row) for idx, row in enumerate(dataset)]

    for future in as_completed(futures):
        count += 1
        if count % batch_size == 0:
            save_data(new_data, output_dir, part)
            new_data = {
                'query': [],
                'text': [],
                'raw_id': []
            }
            part += 1

if new_data['query']:
    save_data(new_data, output_dir, part)

progress_bar.close()

print("Processing complete.")

print(f"Dataset saved to {output_dir}")
# print(f"Loaded dataset sample: {new_dataset[0]}")
# print(f"Number of samples loaded: {len(new_dataset)}")

# #print random 6 samples
# print("Random 6 samples:")
# import random
# for i in range(6):
#     random.seed()
#     seed = random.randint(0, len(new_dataset))
#     print(f"Query: {new_dataset[seed]['query']}")
#     print(f"Positive Image: {new_dataset[seed]['text']}")
#     print(f"Positive Index: {new_dataset[seed]['raw_id']}")
#     print("\n")