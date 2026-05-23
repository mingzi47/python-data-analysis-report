# 06 — 验证主键唯一性

## 描述
检查三个核心表的主键唯一性，确认数据一致性。

## 依赖
- `02-load-games`, `03-load-users`, `04-load-recommendations`

## 输入
- `games_df`, `users_df`, `recommendations_df`

## 输出
- 打印验证结果（无返回值）

## 步骤
1. `games_df['app_id'].is_unique` 检查游戏 ID 唯一性；若有重复，打印重复的 `app_id` 和数量
2. `users_df['user_id'].is_unique` 检查用户 ID 唯一性
3. `recommendations_df.duplicated(subset=['user_id', 'app_id']).sum()` 检查是否有同一用户对同一游戏的重复评价
4. 如有重复评价，按 `['user_id', 'app_id']` 分组统计重复次数分布

## 验收标准
- 确认 `games.app_id` 是否唯一（预期：唯一）
- 确认 `users.user_id` 是否唯一（预期：唯一）
- 确认 `recommendations.(user_id, app_id)` 是否有重复（预期：可能有少数重复，后续清洗处理）
