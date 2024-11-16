import pandas as pd
import numpy as np
import re

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

# 剔除位置异常的记录，如位置字段为空或包含乱码/n/a的记录
invalid_location_mask = users['country'].isnull() | users['state'].isnull() | users['city'].isnull() | users['country'].str.contains('n/a|unknown|[^\x00-\x7F]', case=False) | users['state'].str.contains('n/a|unknown|[^\x00-\x7F]', case=False) | users['city'].str.contains('n/a|unknown|[^\x00-\x7F]', case=False)
users = users[~invalid_location_mask]

# 选择一万名用户（若总数大于一万）
selected_users = users.sample(n=10000, random_state=1) if len(users) > 10000 else users

# 生成用户集合
user_set = set(selected_users['user_id'])

# 显示生成的经过清洗后的用户特征矩阵和用户集合的大小
print(selected_users.head())
print(f'Total number of selected users: {len(user_set)}')

# 保存经过清洗后的用户特征矩阵为CSV文件
selected_users.to_csv('cleaned_users.csv', index=False)