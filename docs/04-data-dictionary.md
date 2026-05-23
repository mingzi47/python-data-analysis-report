# 04 — 数据字典

## games.csv

游戏元信息表。每行是一款 Steam 游戏或附加内容。

| # | 字段名 | 类型 | 说明 | 示例 |
|---|--------|------|------|------|
| 1 | `app_id` | int | Steam 唯一应用 ID | `13500` |
| 2 | `title` | str | 游戏标题 | `"Prince of Persia: Warrior Within™"` |
| 3 | `date_release` | str/date | 发布日期 | `"2008-11-21"` |
| 4 | `win` | bool | 是否支持 Windows | `true` |
| 5 | `mac` | bool | 是否支持 macOS | `false` |
| 6 | `linux` | bool | 是否支持 Linux | `false` |
| 7 | `rating` | str | Steam 文本评分标签（"Overwhelmingly Positive"、"Very Positive"、"Mixed" 等） | `"Very Positive"` |
| 8 | `positive_ratio` | int | 正面评价百分比（0-100），数值型评分的主力字段 | `84` |
| 9 | `user_reviews` | int | 用户评价总数 | `2199` |
| 10 | `price_final` | float | 最终价格（美元）；若与 `price_original` 一致则表示未打折 | `9.99` |
| 11 | `price_original` | float | 原价（美元） | `9.99` |
| 12 | `discount` | float | 折扣比例（0-1）；0 = 无折扣 | `0.0` |
| 13 | `steam_deck` | bool | Steam Deck 兼容性 | `true` |

> **注意：**
> - `rating` 是文本标签（如 "Very Positive"），**不能直接用于数值计算**。需要数值评分时使用 `positive_ratio`。
> - `positive_ratio` 是 0-100 的整数，表示正面评价百分比。在特征工程中除以 100 映射到 [0, 1] 区间。
> - 实际数据中**没有** `user_score`、`owners` 列，但有 `win`、`mac`、`linux`、`user_reviews` 列。
> - `owners` 列在某些版本的 Steam 数据集中存在但在本项目使用的版本中缺失。

---

## users.csv

匿名用户公开档案表。每行是一位 Steam 用户。

| # | 字段名 | 类型 | 说明 | 示例 |
|---|--------|------|------|------|
| 1 | `user_id` | int | 匿名化用户 ID | `76561197960292822` |
| 2 | `products` | int | 用户已购产品数 | `42` |
| 3 | `reviews` | int | 用户发布的评价总数 | `15` |

> **限制：** 无年龄、性别、地区等人口统计信息。`products` 和 `reviews` 是聚合统计值，无法知道用户具体购买了哪些游戏（除了已在 `recommendations.csv` 中出现过的评价记录）。

---

## recommendations.csv

用户推荐记录表（核心交互数据）。每行是一条用户对游戏的评价，总记录数超过 4,100 万。

| # | 字段名 | 类型 | 说明 | 示例 |
|---|--------|------|------|------|
| 1 | `app_id` | int | 被评价的游戏 ID（外键 → `games.app_id`） | `730` |
| 2 | `user_id` | int | 评价用户 ID（外键 → `users.user_id`） | `76561197960292822` |
| 3 | `is_recommended` | int | 是否推荐：1 = 推荐，0 = 不推荐（目标变量） | `1` |
| 4 | `hours` | float | 用户游玩时长（小时） | `1234.5` |
| 5 | `date` | str/date | 评价发布时间 | `"2018-03-15"` |
| 6 | `helpful` | int | 该评价被其他用户标记为"有帮助"的次数 | `12` |
| 7 | `funny` | int | 该评价被其他用户标记为"有趣"的次数 | `3` |
| 8 | `review_id` | int | 评价唯一 ID | `24251963` |

> **注意：**
> - `hours` 是重要的补充特征：用户玩的时间越长，评价的可信度/权重可能越高
> - `helpful` 和 `funny` 反映社区对评价本身的认可度，可用于权重调整
> - `(user_id, app_id)` 理论上是唯一的，但可能存在历史修改记录（新旧评价并存），需按 `date` 去重保留最新
> - 实际数据有 `review_id` 列，为评价的唯一标识符

---

## games_metadata.json

游戏富文本元数据（非表格化 JSON，每行一个独立 JSON 对象）。

| # | 字段名 | 类型 | 说明 | 示例 |
|---|--------|------|------|------|
| 1 | `app_id` | int | Steam 唯一应用 ID（外键 → `games.app_id`） | `13500` |
| 2 | `description` | str | 游戏描述文本（多语言混合） | `"Enter the dark underworld of..."` |
| 3 | `tags` | list[str] | 用户自定义标签数组 | `["Action", "Adventure", "Parkour", "Singleplayer", ...]` |

> **注意：**
> - **实际数据仅包含以上 3 个字段。** 不包含 `genres`、`type`、`early_access` 列（这些在文档描述的版本中存在但在本项目使用的数据版本中缺失）。
> - `tags` 是用户自定义的，数量多、粒度细、有噪声（如 "Masterpiece"、"Walking Simulator"），通常有 10-20 个标签。
> - `tags` 在项目中被同时用作"标签"和"类型"的代理——当它被 explode 展开时能提供类似 genres 的分类信息。
> - `description` 是多语言文本，初版分析可以不使用，后续可尝试 NLP 特征。

---

## 字段使用优先级

按在分析中的必要性分三级：

### 核心字段（必须使用）

| 数据集 | 字段 | 用途 |
|--------|------|------|
| games | `app_id`, `title`, `positive_ratio`, `rating`, `price_final`, `date_release` | 游戏基础画像 + 建模特征 |
| users | `user_id`, `products`, `reviews` | 用户行为画像 + 建模特征 |
| recommendations | `user_id`, `app_id`, `is_recommended`, `hours`, `date` | 交互数据 + 目标变量 |
| metadata | `app_id`, `tags` | 标签特征（代理 genres 功能） |

### 扩展字段（可用但非必需）

| 数据集 | 字段 | 用途 |
|--------|------|------|
| games | `discount`, `steam_deck`, `win`, `mac`, `linux` | 可选的游戏侧特征（平台兼容性可编码） |
| games | `user_reviews` | 游戏的评价总数，可替代 owners 的流行度信号 |
| recommendations | `helpful`, `funny` | 评价权重调整 |
| recommendations | `review_id` | 评价唯一标识（去重备选） |

### 待选字段（初版暂不使用）

| 数据集 | 字段 | 原因 |
|--------|------|------|
| metadata | `description` | 多语言文本，NLP 成本高 |
