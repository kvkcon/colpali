import numpy as np
from datasets import load_from_disk, load_dataset
# from PIL import Image
from tqdm import tqdm 

def dcg_at_k(relevance, k=5, method=0):
    """
    计算DCG@k
    :param relevance: 相关性分数列表
    :param k: 返回的项数
    :param method: 计算方法，0或1
    :return: DCG@k的值
    """
    if method == 0:
        gain = [pow(2, rel) - 1.0 for rel in relevance[:k]]
    elif method == 1:
        gain = [pow(2, rel) / np.log2(i + 2) for i, rel in enumerate(relevance[:k])]
    else:
        assert False, 'invalid method'
    
    return np.sum(gain)

def ndcg_at_k(relevance, scores, k=5):
    """
    计算NDCG@k
    :param relevance: 相关性分数列表
    :param scores: 模型给出的分数列表
    :param k: 返回的项数
    :return: NDCG@k的值
    """
    # 计算DCG@k
    dcg_max = dcg_at_k(sorted(relevance, reverse=True), k)
    # 计算实际的DCG@k
    actual_dcg = dcg_at_k(scores, k)
    
    # 计算NDCG@k
    ndcg = actual_dcg / dcg_max
    return ndcg

def main():
    """queries = [
        {
            'query': 'What does the function animateComputerMoving do?',
            'positive_index': 877,
            'topk_indices': [877, 2307, 1528, 1646, 3074, 967, 4961, 572, 2566, 744],
            'scores': [0.7172300815582275, 0.6409473419189453, 0.6395168304443359, 0.6331759691238403, 0.6331455111503601, 0.6257294416427612, 0.6155081987380981, 0.6135450601577759, 0.6135210394859314, 0.6120305061340332]
        },
        # ... 其他查询
    ]"""

    output_dir = "/data/hf_datasets/ocr_t2s_train_set_colpali_jina3_train5000_4096/"#rerank_train_set_colpali_1_2merge_test50 #/data/hf_datasets/ocr_erank_train_set_colpali_1_2merge_test500_multi/part_1
    loaded_dataset = load_from_disk(output_dir)
    print("loaded_dataset",load_dataset)
    # loaded_dataset = load_dataset(output_dir, split="train[:50]")
    print(f"Loaded dataset sample: {loaded_dataset[0]}")
    print(f"Number of samples loaded: {len(loaded_dataset)}")
    queries = loaded_dataset

    ndcg_sum = 0.0
    ndcg_min = 1.0
    ndcg_max = 0.0
    # count every query NDCG@k
    for i in tqdm(range(len(queries)), desc="Processing queries..."):
        relevance = [1.0] 
        scores = [sc/10 for sc in queries['scores'][i]]
        k = min(len(scores), 10)  
        ndcg = ndcg_at_k(relevance, scores, k)
        ndcg_sum += ndcg
        ndcg_min = min(ndcg_min, ndcg)
        ndcg_max = max(ndcg_max, ndcg)
        # print(f"Query: {query['query']}")
        # print(f"NDCG@{k}: {ndcg}")
        # print("\n")
    print("\n")
    print(f"mNDCG@{k}: {ndcg/len(loaded_dataset)}")
    print(f"mNDCG@{k} min: {ndcg_min}")
    print(f"mNDCG@{k} max: {ndcg_max}")

if __name__ == "__main__":
    main()