# 04 — 去重

## 描述
去除 `recommendations.csv` 中同一用户对同一游戏的重复评价（保留最新），以及 `games.csv` 中的重复游戏。

## 依赖
- `02-type-conversion`：需要日期已转换

## 输入
- `recommendations_df`, `games_df`

## 输出
- 去重后的 DataFrame

## 步骤
1. `recommendations_df` 按 `['user_id', 'app_id']` 分组，保留每组中 `date` 最新的记录
2. 打印去重前后的行数差异
3. `games_df` 按 `app_id` 去重，保留第一条
4. 确认 `users_df` 无重复（若有则去重）

## 验收标准
- 去重后 `recommendations.(user_id, app_id)` 组合唯一
- 去重后 `games.app_id` 唯一
- 去重丢弃的行数记录在日志中
