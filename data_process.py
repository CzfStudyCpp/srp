import pandas as pd
import numpy as np
from itertools import combinations
import os
from scipy.sparse import coo_matrix, save_npz, load_npz
from joblib import Parallel, delayed, parallel_backend
from multiprocessing import Pool
import time
from random import sample

start_time = time.time()


# Step 1: 设置输入和输出路径
input_dir = './input_data'
output_dir = './similarity_matrices'
os.makedirs(output_dir, exist_ok=True)

# Step 2: 加载数据
print("加载清洗后的数据...")
ratings = pd.read_csv(os.path.join(input_dir, 'ratings_matrix.csv'))
rated_users = pd.read_csv(os.path.join(input_dir, 'rated_users.csv'))['user_id'].tolist()
rated_items = pd.read_csv(os.path.join(input_dir, 'rated_items.csv'))['isbn'].tolist()

# 创建用户和项目的索引映射（用于稀疏矩阵的索引）
user_index = {user_id: idx for idx, user_id in enumerate(rated_users)}
item_index = {item_id: idx for idx, item_id in enumerate(rated_items)}

np.random.seed(42)
# 定义余弦相似度函数（用户相似度）
def cosine_similarity(user_ratings_1, user_ratings_2, common_items):
    if len(common_items) == 0:
        return 0
    ratings_1 = user_ratings_1.loc[common_items].values
    ratings_2 = user_ratings_2.loc[common_items].values
    numerator = np.dot(ratings_1, ratings_2)
    denominator = np.linalg.norm(ratings_1) * np.linalg.norm(ratings_2)
    return numerator / denominator if denominator != 0 else 0

# 定义皮尔森相关系数函数（项目相似度）
def pearson_similarity(item_ratings_1, item_ratings_2, common_users):
    if len(common_users) == 0:
        return 0
    ratings_1 = item_ratings_1.loc[common_users].values
    ratings_2 = item_ratings_2.loc[common_users].values
    mean_1, mean_2 = np.mean(ratings_1), np.mean(ratings_2)
    numerator = np.sum((ratings_1 - mean_1) * (ratings_2 - mean_2))
    denominator = np.sqrt(np.sum((ratings_1 - mean_1) ** 2)) * np.sqrt(np.sum((ratings_2 - mean_2) ** 2))
    return numerator / denominator if denominator != 0 else 0

# 过滤有效用户对
def filter_valid_user_pairs(user_pairs, user_ratings):
    return [
        (u1, u2) for u1, u2 in user_pairs
        if u1 in user_ratings.index and u2 in user_ratings.index
        and any(user_ratings.loc[u1].notna() & user_ratings.loc[u2].notna())
    ]

# 过滤有效项目对
def filter_valid_item_pairs(item_pairs, item_ratings):
    return [
        (i1, i2) for i1, i2 in item_pairs
        if i1 in item_ratings.index and i2 in item_ratings.index
        and any(item_ratings.loc[i1].notna() & item_ratings.loc[i2].notna())
    ]

def compute_user_similarity(user_pairs, user_ratings):
    with parallel_backend('threading', n_jobs=-1):
        results = Parallel()(
            delayed(cosine_similarity)(
                user_ratings.loc[u1], user_ratings.loc[u2],
                user_ratings.columns[
                    user_ratings.loc[u1].notna() & user_ratings.loc[u2].notna()
                ]
            )
            for u1, u2 in user_pairs
        )
    return [(u1, u2, sim) for (u1, u2), sim in zip(user_pairs, results)]

def compute_item_similarity(item_pairs, item_ratings):
    with parallel_backend('threading', n_jobs=-1):
        results = Parallel()(
            delayed(pearson_similarity)(
                item_ratings.loc[i1], item_ratings.loc[i2],
                item_ratings.columns[
                    item_ratings.loc[i1].notna() & item_ratings.loc[i2].notna()
                ]
            )
            for i1, i2 in item_pairs
        )
    return [(i1, i2, sim) for (i1, i2), sim in zip(item_pairs, results)]

# 保存矩阵的稀疏格式
def save_sparse_matrix(output_file, sparse_matrix):
    save_npz(output_file, sparse_matrix.tocsr())
    print(f"稀疏矩阵已保存至 {output_file}")

# Step 3: 分批计算相似度
def process_batch(batch_id):
    print(f"开始处理批次 {batch_id} ...")


    # 随机选择用户和项目
    current_rated_users = np.random.choice(rated_users, size=min(batch_size, len(rated_users)), replace=False)
    current_rated_items = np.random.choice(rated_items, size=min(batch_size, len(rated_items)), replace=False)

    # 提取当前用户和项目的评分数据
    user_ratings = ratings[ratings['user_id'].isin(current_rated_users)].pivot(index='user_id', columns='isbn', values='book_rating')
    item_ratings = ratings[ratings['isbn'].isin(current_rated_items)].pivot(index='isbn', columns='user_id', values='book_rating')

    # 过滤有效用户对和项目对
    user_pairs = list(combinations(current_rated_users, 2))
    item_pairs = list(combinations(current_rated_items, 2))
    user_pairs = filter_valid_user_pairs(user_pairs, user_ratings)
    item_pairs = filter_valid_item_pairs(item_pairs, item_ratings)

    # 并行计算用户相似度
    print(f"批次 {batch_id}: 开始用户相似度计算...")
    user_sim_results = compute_user_similarity(user_pairs, user_ratings)

    # 并行计算项目相似度
    print(f"批次 {batch_id}: 开始项目相似度计算...")
    item_sim_results = compute_item_similarity(item_pairs, item_ratings)

    return user_sim_results, item_sim_results

print("开始分批处理相似度...")
batch_size = 1000
num_batches = 3
threshold = 0.01  # 设置相似度阈值

# 稀疏矩阵动态构建
user_sim_values, user_sim_rows, user_sim_cols = [], [], []
item_sim_values, item_sim_rows, item_sim_cols = [], [], []

# 使用多进程进行批次处理
with Pool(processes=os.cpu_count()) as pool:
    batch_results = pool.map(process_batch, range(num_batches))

# 收集结果并动态构建稀疏矩阵
for user_sim_results, item_sim_results in batch_results:
    for u1, u2, sim in user_sim_results:
        if abs(sim) > threshold:  # 仅保留高于阈值的结果
            user_sim_rows.append(user_index[u1])
            user_sim_cols.append(user_index[u2])
            user_sim_values.append(sim)
    for i1, i2, sim in item_sim_results:
        if abs(sim) > threshold:  # 仅保留高于阈值的结果
            item_sim_rows.append(item_index[i1])
            item_sim_cols.append(item_index[i2])
            item_sim_values.append(sim)

# 动态生成稀疏矩阵
user_sim_matrix = coo_matrix((user_sim_values, (user_sim_rows, user_sim_cols)),
                              shape=(len(rated_users), len(rated_users)))
item_sim_matrix = coo_matrix((item_sim_values, (item_sim_rows, item_sim_cols)),
                              shape=(len(rated_items), len(rated_items)))


# 确保矩阵对称性
user_sim_matrix = user_sim_matrix + user_sim_matrix.T
item_sim_matrix = item_sim_matrix + item_sim_matrix.T
# Step 4: 保存最终的用户相似度矩阵和项目相似度矩阵
print("保存相似度矩阵...")
save_sparse_matrix(os.path.join(output_dir, 'user_similarity_matrix_optimized.npz'), user_sim_matrix)
save_sparse_matrix(os.path.join(output_dir, 'item_similarity_matrix_optimized.npz'), item_sim_matrix)

print(f"用户相似度矩阵和项目相似度矩阵已保存到 {output_dir}.")

# 主代码逻辑
print(f"处理完成，总耗时: {time.time() - start_time:.2f} 秒")

# 检查生成的矩阵
for file in os.listdir(output_dir):
    if file.endswith('.npz'):
        matrix = load_npz(os.path.join(output_dir, file))
        print(f"文件: {file}")
        print(f"矩阵维度: {matrix.shape}")
        print(f"非零元素数量: {matrix.nnz}")
        print(f"稀疏度: {1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4f}")
