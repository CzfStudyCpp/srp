import os

from scipy.sparse import load_npz
output_dir = './similarity_matrices'
# 加载用户和项目的相似度矩阵
user_sim_matrix = load_npz(os.path.join(output_dir, 'user_similarity_matrix_optimized.npz'))
item_sim_matrix = load_npz(os.path.join(output_dir, 'item_similarity_matrix_optimized.npz'))

# 查看矩阵内容
print(user_sim_matrix.shape)  # 输出矩阵维度
print(item_sim_matrix.nnz)    # 输出非零元素数量
