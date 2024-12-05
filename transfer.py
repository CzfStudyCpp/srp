import os
import pandas as pd
from scipy.sparse import load_npz

# 设置输入和输出路径
input_dir = './similarity_matrices'
output_dir = './output_files'
os.makedirs(output_dir, exist_ok=True)

def save_to_csv(sparse_matrix, output_file, user_or_item_list):
    """
    将稀疏矩阵转化为直观的 CSV 文件。
    """
    rows, cols = sparse_matrix.nonzero()
    data = sparse_matrix.data
    results = []
    for r, c, d in zip(rows, cols, data):
        results.append({
            "Entity 1": user_or_item_list[r],
            "Entity 2": user_or_item_list[c],
            "Similarity": d
        })
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"保存到 CSV 文件: {output_file}")

def save_to_json(sparse_matrix, output_file, user_or_item_list):
    """
    将稀疏矩阵转化为直观的 JSON 文件。
    """
    rows, cols = sparse_matrix.nonzero()
    data = sparse_matrix.data
    results = []
    for r, c, d in zip(rows, cols, data):
        results.append({
            "Entity 1": user_or_item_list[r],
            "Entity 2": user_or_item_list[c],
            "Similarity": d
        })
    df = pd.DataFrame(results)
    df.to_json(output_file, orient='records', lines=True)
    print(f"保存到 JSON 文件: {output_file}")

# 读取用户和项目相似度矩阵
print("加载生成的稀疏矩阵...")
user_similarity_matrix = load_npz(os.path.join(input_dir, 'user_similarity_matrix_optimized.npz'))
item_similarity_matrix = load_npz(os.path.join(input_dir, 'item_similarity_matrix_optimized.npz'))

# 加载用户和项目列表
rated_users = pd.read_csv('./input_data/rated_users.csv')['user_id'].tolist()
rated_items = pd.read_csv('./input_data/rated_items.csv')['isbn'].tolist()

# 保存用户相似度结果
save_to_csv(user_similarity_matrix, os.path.join(output_dir, 'user_similarity.csv'), rated_users)
save_to_json(user_similarity_matrix, os.path.join(output_dir, 'user_similarity.json'), rated_users)

# 保存项目相似度结果
save_to_csv(item_similarity_matrix, os.path.join(output_dir, 'item_similarity.csv'), rated_items)
save_to_json(item_similarity_matrix, os.path.join(output_dir, 'item_similarity.json'), rated_items)

print("结果已保存到 ./output_files/")
