# 02 — 构造用户侧特征

## 描述
从 `users.csv` 提取用户侧特征。

## 依赖
- 数据清洗完成

## 输入
- `users_df`（含 `user_id`, `products`, `reviews`）

## 输出
- `user_features_df`，含以下列：
  - `user_products_count` (int)
  - `user_reviews_count` (int)
  - `review_ratio` (float)：`reviews / products`

## 步骤
1. 直接使用 `products` → `user_products_count`
2. 直接使用 `reviews` → `user_reviews_count`
3. 计算 `review_ratio = reviews / max(products, 1)`（避免除零）
4. 确保索引为 `user_id`

## 验收标准
- 无缺失值
- `review_ratio` 范围合理（0-1+，可能有极少 >1 的异常）
- 索引为 `user_id`
