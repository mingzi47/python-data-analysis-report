# 05 — 头部集中度分析

## 描述
分析游戏推荐量的集中度：少数头部游戏是否占据了大多数推荐？计算长尾分布特征。

## 依赖
- 数据清洗完成（需要 `recommendations_df`）

## 输入
- `recommendations_df`

## 输出
- 长尾分布曲线（`outputs/figures/long_tail.png`）+ Gini 系数

## 步骤
1. 按 `app_id` 统计推荐量：`review_counts = df.groupby('app_id').size()`
2. 降序排列，计算累计占比：`cumsum / total`
3. 画 log-log 散点图：x = 排名，y = 推荐量
4. 计算 Gini 系数：衡量推荐量的不平等程度
5. 计算 Top 1%, 5%, 10%, 20% 游戏占据的推荐量占比

## 验收标准
- Gini 系数已计算（预期 > 0.5，表明高度集中）
- Top N% 的集中度数据清晰
- 长尾分布曲线展示完整
