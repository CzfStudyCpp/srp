import pandas as pd
import numpy as np
import re
import os
from itertools import combinations

# Step 1: 读取Book-Crossing用户信息数据集
# 使用pandas的read_csv函数来读取用户数据
users = pd.read_csv('./BX-Users.csv', sep=';', encoding="latin-1", on_bad_lines='skip')

# 设置最大展示列数量
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Step 2: 数据清洗
# 处理列名，去除空格并转换为小写
users.columns = users.columns.str.strip().str.lower().str.replace('-', '_')

# 清理位置列中的多余空格
users['location'] = users['location'].str.replace('\s+', ' ', regex=True).str.strip()

# 将位置特征划分为国家，省，市三个特征，确保始终有三列
user_location_expanded = users['location'].str.split(',', n=2, expand=True)
user_location_expanded.columns = ['city', 'state', 'country']

# 处理拆分后得到的 NaN 值（如果有些记录少于三部分）
user_location_expanded['state'] = user_location_expanded['state'].str.strip().fillna('unknown')
user_location_expanded['country'] = user_location_expanded['country'].str.strip().fillna('unknown')
user_location_expanded['city'] = user_location_expanded['city'].str.strip().fillna('unknown')

# 将清洗后的位置数据加入原始数据框
users = users.join(user_location_expanded)
users.drop(columns=['location'], inplace=True)

# 将空字符串置空
users['country'] = users['country'].replace('', np.nan)
users['state'] = users['state'].replace('', np.nan)
users['city'] = users['city'].replace('', np.nan)

# 剔除异常年龄值并剔除年龄为空的记录
users = users[(users.age >= 4) & (users.age <= 105)]

# 剔除位置异常的记录，如位置字段为空或包含乱码/n/a/HTML字符引用的记录
invalid_location_mask = users['country'].isnull() | users['state'].isnull() | users['city'].isnull() | \
                      users['country'].str.contains('n/a|unknown|[^\x00-\x7F]|&#[0-9]+;', case=False) | \
                      users['state'].str.contains('n/a|unknown|[^\x00-\x7F]|&#[0-9]+;', case=False) | \
                      users['city'].str.contains('n/a|unknown|[^\x00-\x7F]|&#[0-9]+;', case=False)
users = users[~invalid_location_mask]

# 进一步清洗，移除所有包含 HTML 字符实体或者非 ASCII 字符的记录
users = users[~users['country'].str.contains('&#|[^\x00-\x7F]', case=False, na=False)]
users = users[~users['state'].str.contains('&#|[^\x00-\x7F]', case=False, na=False)]
users = users[~users['city'].str.contains('&#|[^\x00-\x7F]', case=False, na=False)]

# 创建结果文件的子目录
output_dir = './input_data'  # 修改输出路径

os.makedirs(output_dir, exist_ok=True)

# 生成用户特征矩阵（DataFrame形式）
user_matrix = users.pivot_table(index='user_id', values=['age', 'city', 'state', 'country'], aggfunc='first')

# 显示生成的经过清洗后的用户特征矩阵
print(users.head())
print(f'Total number of users after cleaning: {len(users)}')
print(user_matrix.head())

# 保存经过清洗后的用户特征矩阵为CSV文件
cleaned_users_path = os.path.join(output_dir, 'cleaned_users.csv')
user_matrix_path = os.path.join(output_dir, 'user_feature_matrix.csv')
users.to_csv(cleaned_users_path, index=False)
user_matrix.to_csv(user_matrix_path, index=True)

# Step 3: 读取Book-Crossing图书信息数据集
# 使用pandas的read_csv函数来读取图书数据
books = pd.read_csv('./BX_Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')

# 处理列名，去除空格并转换为小写
books.columns = books.columns.str.strip().str.lower().str.replace('-', '_')

# 清理图书特征中的多余空格
books['book_title'] = books['book_title'].str.strip()
books['book_author'] = books['book_author'].str.strip()
books['publisher'] = books['publisher'].str.strip()

# 剔除包含乱码或异常值的记录，如HTML字符引用或非ASCII字符
invalid_books_mask = books['book_title'].str.contains('[^\x00-\x7F]|&#[0-9]+;', case=False) | \
                     books['book_author'].str.contains('[^\x00-\x7F]|&#[0-9]+;', case=False) | \
                     books['publisher'].str.contains('[^\x00-\x7F]|&#[0-9]+;', case=False)
books = books[~invalid_books_mask]

# 进一步清洗，移除所有包含 HTML 字符实体或者非 ASCII 字符的记录
books = books[~books['book_title'].str.contains('&#|[^\x00-\x7F]', case=False, na=False)]
books = books[~books['book_author'].str.contains('&#|[^\x00-\x7F]', case=False, na=False)]
books = books[~books['publisher'].str.contains('&#|[^\x00-\x7F]', case=False, na=False)]

# 移除图书数据中的图像链接列
books.drop(columns=['image_url_s', 'image_url_m', 'image_url_l'], inplace=True)

# 保存经过清洗后的图书特征矩阵
cleaned_books_path = os.path.join(output_dir, 'cleaned_books.csv')
books.to_csv(cleaned_books_path, index=False)

# 显示生成的经过清洗后的图书特征矩阵
print(books.head())

# 生成项目特征矩阵（DataFrame形式）
item_matrix = books.pivot_table(index='isbn', values=['book_title', 'book_author', 'year_of_publication', 'publisher'], aggfunc='first')

# 保存项目特征矩阵为CSV文件
item_matrix_path = os.path.join(output_dir, 'item_feature_matrix.csv')
item_matrix.to_csv(item_matrix_path, index=True)

# Step 4: 读取Book-Crossing评分数据集
# 使用pandas的read_csv函数来读取评分数据
# 读取和清洗评分数据集
ratings = pd.read_csv('./BX-Book-Ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
ratings.columns = ratings.columns.str.strip().str.lower().str.replace('-', '_')
ratings = ratings[ratings['book_rating'] > 0]

# 获取评分矩阵中已评分的项目集合 I 和用户集合 U（忽略隐含评分，取评分 > 0 的记录）
rated_items = ratings[ratings['book_rating'] > 0]['isbn'].unique()
rated_users = ratings[ratings['book_rating'] > 0]['user_id'].unique()


ratings.to_csv(os.path.join(output_dir, 'ratings_matrix.csv'), index=False)

# 保存用户集合和项目集合
np.save(os.path.join(output_dir, 'rated_items.npy'), rated_items)
np.save(os.path.join(output_dir, 'rated_users.npy'), rated_users)

pd.DataFrame(rated_users, columns=['user_id']).to_csv(os.path.join(output_dir, 'rated_users.csv'), index=False)
pd.DataFrame(rated_items, columns=['isbn']).to_csv(os.path.join(output_dir, 'rated_items.csv'), index=False)



