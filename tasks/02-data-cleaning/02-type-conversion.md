# 02 — 数据类型转换

## 描述
统一转换日期、价格、目标变量等列的类型。

## 依赖
- `01-missing-values`：需要在缺失值处理后进行

## 输入
- `games_df`, `recommendations_df`

## 输出
- 类型转换后的 DataFrame

## 步骤
1. `games_df['date_release']` → `pd.to_datetime()`，容错 `errors='coerce'`
2. 从 `date_release` 提取 `release_year`, `release_month`
3. `games_df['price_final']` → `float`（确认已为数值类型）
4. `recommendations_df['is_recommended']` → `int`（0/1）
5. `recommendations_df['date']` → `pd.to_datetime()`
6. 打印各列转换后的 `dtype` 确认

## 验收标准
- `release_year` 和 `release_month` 列已生成
- `price_final` 为 float，`is_recommended` 为 int
- 日期转换失败的记录（NaT）占比 < 1%
