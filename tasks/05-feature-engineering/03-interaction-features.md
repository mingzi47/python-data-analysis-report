# 03 — 构造交互特征

## 描述
从 `recommendations.csv` 按用户和游戏聚合出交互特征。这是关键步骤——必须只在训练集上聚合避免数据泄漏。

## 依赖
- 数据清洗完成

## 输入
- `recommendations_df`（含 `user_id`, `app_id`, `is_recommended`, `hours`, `helpful`, `funny`）

## 输出
- 交互特征列，合并到主记录中：
  - `game_review_count` (int)：每个游戏的总评价数
  - `game_recommend_rate` (float)：每个游戏的推荐率
  - `game_avg_hours` (float)：每个游戏的平均游玩时长
  - `user_review_count` (int)：每个用户的总评价数
  - `user_recommend_rate` (float)：每个用户的推荐率
  - `user_avg_hours` (float)：每个用户的平均游玩时长

## 步骤
1. 按 `app_id` 聚合：`count`, `is_recommended.mean()`, `hours.mean()`
2. 按 `user_id` 聚合：`count`, `is_recommended.mean()`, `hours.mean()`
3. 将聚合结果 merge 回 `recommendations_df`
4. **重要：** 记录在代码注释中：最终流水线中，这些聚合必须在 `split_data` 之后的训练集上计算，再映射到测试集

## 验收标准
- 聚合特征无缺失（merge 率 ~100%）
- `game_recommend_rate` 范围 [0, 1]
- 数据泄漏风险已在注释中标注
