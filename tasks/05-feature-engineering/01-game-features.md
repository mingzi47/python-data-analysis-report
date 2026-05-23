# 01 — 构造游戏侧特征

## 描述
从 `games.csv` 和 `games_metadata.json` 提取游戏侧特征。

## 依赖
- 数据清洗完成（含 metadata 合并）

## 输入
- `games_df`（含 `price_final`, `rating`, `release_year`, `tags`, `genres`, `type`, `early_access`）

## 输出
- `game_features_df`，含以下列：
  - `price` (float)：最终价格
  - `is_free` (int 0/1)：是否免费
  - `rating` (float)：评分
  - `years_since_release` (int)：距今年数
  - `num_tags` (int)：标签数量
  - `num_genres` (int)：类型数量
  - `early_access` (int 0/1)：是否抢先体验
  - 热门 genres 的 one-hot 列（出现频率 > 1% 的类型）
  - 热门 tags 的 one-hot 列（出现频率 > 1% 的标签）

## 步骤
1. 提取基础数值特征：`price`, `is_free`, `rating`, `years_since_release`
2. 计算 `num_tags = df['tags'].apply(len)`, `num_genres = df['genres'].apply(len)`
3. 选取出现频率 > 1% 的 genres，用 `MultiLabelBinarizer` 展开为 one-hot 列
4. 选取出现频率 > 1% 的 tags，同样展开
5. 确保索引为 `app_id`

## 验收标准
- 数值特征列无缺失
- one-hot 列数合理（genres: 10-20 列, tags: 20-50 列）
- 索引为 `app_id`，方便后续 join
