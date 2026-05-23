# 03 — 游戏类型频率分析

## 描述
统计 tags 和 genres 的出现频率，识别最热门的游戏类型和类型组合。

## 依赖
- 数据清洗完成（`05-merge-metadata` 后 `games_df` 含 `tags`, `genres` 列）

## 输入
- `games_df`（含 `tags` 和 `genres` list 列）

## 输出
- Top 20 条形图（`outputs/figures/genre_frequency.png`）+ 统计表

## 步骤
1. 展平 `genres` 列：`df.explode('genres')['genres'].value_counts()`
2. 展平 `tags` 列：同上，取 Top 30
3. 分别画横向条形图（`sns.barplot`），genres 和 tags 各一张
4. 统计每个游戏的平均标签数（`df['tags'].apply(len).describe()`）
5. 分析常见类型组合：取 Top 5 genres，统计它们的共现矩阵

## 验收标准
- Top 类型列表有业务可读性
- 每游戏平均标签数合理
- 共现矩阵能反映常见的类型组合模式（如 "Indie + Action"）
