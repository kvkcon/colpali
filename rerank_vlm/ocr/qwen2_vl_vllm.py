from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from datasets import Dataset, load_dataset, Features, Value, Sequence, Image, load_from_disk

from typing import List, cast, Optional, Tuple, Union
import yaml
import torch
import os
import gc
from tqdm import tqdm

def save_data(new_data, output_dir, part):
    part_output_dir = os.path.join(output_dir, f'part_{part}')
    os.makedirs(part_output_dir, exist_ok=True)
    part_dataset = Dataset.from_dict(new_data)
    part_dataset.save_to_disk(part_output_dir)
    print(f"Part {part} saved to {part_output_dir}")
    new_data.clear()
    # gc.collect() 


def main():
    with open('/home/jinaai/binwenbai/rerank_vlm/config/ocr/test_part0.yaml', 'r') as f:
        config = yaml.safe_load(f)

    MODEL_PATH = config['model']['name'] #"/data/hf_models/Qwen2-VL-7B-Instruct/"
    tensor_parallel_size = config['model']['tensor_parallel_size']

    llm = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"image": 10, "video": 10},
        tensor_parallel_size=tensor_parallel_size,#tensor_parallel_size=4
        # max_num_batched_tokens=4096,
        # enable_chunked_prefill=True,
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=config['model']['max_tokens'],
        stop_token_ids=[],

    )

    dataset = cast(Dataset, load_dataset(config['dataset']['input_path'], split=config['dataset']['split']))
    print("Dataset loaded: ", dataset)

    # For video input, you can pass following values instead:
    # "type": "video",
    # "video": "<video URL>",

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    new_data = {
        'query': [],
        'text': [],
        'raw_id': []
    }

    output_dir = config['dataset']['output_dir']
    count = 0
    batch_size = config['save']['save_batch_size']
    part = config['save']['split_num']
    for idx,item in enumerate(tqdm(dataset, desc='Processing images')):
        # if idx < 41:
        #     continue
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": item['image'],
                        "min_pixels": 224 * 224,
                        "max_pixels": 1280 * 28 * 28,
                    },
                    {"type": "text", "text": "Extract the structured output of OCR"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # print(f"Generated Text: {generated_text}")


        new_data['query'].append(item['query'])
        new_data['text'].append(generated_text)
        new_data['raw_id'].append(idx+config['save']['split_num'])
        # print(f"Query: {item['query']}")
        # print(f"Raw ID: {idx}")
        # print(outputs[0].outputs[0])
        count += 1
        if count % batch_size == 0:
            save_data(new_data, output_dir, part)
            new_data = {
                'query': [],
                'text': [],
                'raw_id': []
            }
            part += 1
        # break
    if new_data['query']:
        save_data(new_data, output_dir, part)
    print(generated_text)

if __name__ == '__main__':
    main()