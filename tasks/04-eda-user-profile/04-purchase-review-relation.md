# 04 — 购买-评论关系分析

## 描述
分析已购产品数与已发评论数的关系，识别不同行为模式的用户群。

## 依赖
- 数据清洗完成

## 输入
- `users_df`（含 `products`, `reviews` 列）

## 输出
- 散点图（`outputs/figures/purchase_vs_reviews.png`）+ 相关性分析

## 步骤
1. 计算 Spearman 秩相关系数 `products` vs `reviews`
2. 画散点图（log-log 坐标），叠加 y=x 参考线
3. 计算 `reviews / products` 比率（评论/购买比），分析其分布
4. 识别异常模式：买很多评很少、买很少每条都评
5. 按 `reviews/products` 比率分层，统计各层用户特征

## 验收标准
- 相关性量化（预期：中等正相关）
- 评论/购买比的分布有意义
- 异常行为模式有描述
