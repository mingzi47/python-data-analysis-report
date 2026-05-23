# 01 — 处理缺失值

## 描述
统计三张表的缺失值，按预定策略处理。

## 依赖
- `02-load-games`, `03-load-users`, `04-load-recommendations`

## 输入
- `games_df`, `users_df`, `recommendations_df`

## 输出
- 清洗后的三个 DataFrame（缺失值已处理）

## 步骤
1. 对每张表执行 `df.isnull().sum() / len(df)` 计算各列缺失比例
2. `games_df`：删除 `title` 缺失的行；`rating` 缺失用中位数填充；`date_release` 缺失用中位数年份填充
3. `users_df`：`products` 或 `reviews` 缺失填充为 0
4. `recommendations_df`：删除 `is_recommended` 缺失的行
5. 打印处理前后的行数对比

## 验收标准
- 处理后三张表均无关键字段缺失
- 删除的行数在日志中有记录
- `is_recommended` 无 NaN
