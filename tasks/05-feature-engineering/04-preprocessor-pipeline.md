# 04 — 构建特征处理流水线

## 描述
使用 sklearn 的 `ColumnTransformer` 构建可复用的预处理流水线，处理数值标准化和二值特征。

## 依赖
- `01-game-features`, `02-user-features`, `03-interaction-features`

## 输入
- 特征矩阵 `X`（包含所有原始特征列）

## 输出
- `ColumnTransformer` 对象，可直接用于 `fit_transform` / `transform`

## 步骤
1. 将特征列分为两组：
   - 数值特征：`price`, `rating`, `years_since_release`, `num_tags`, `num_genres`, `user_products_count`, `user_reviews_count`, `review_ratio`, `game_review_count`, `game_recommend_rate`, `game_avg_hours`, `user_review_count`, `user_recommend_rate`, `user_avg_hours`
   - 二值特征：`is_free`, `early_access` + 所有 one-hot 的 genre/tag 列
2. 数值特征 → `StandardScaler`
3. 二值特征 → `passthrough`
4. 组合为 `ColumnTransformer`

## 验收标准
- `ColumnTransformer` 能成功 `fit_transform` 训练集
- 处理后的特征矩阵无 NaN
- 特征维度等于输入维度（无信息丢失）
