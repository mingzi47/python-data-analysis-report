# 04 — 数据字典

## games.csv

游戏元信息表。每行是一款 Steam 游戏或附加内容。

| # | 字段名 | 类型 | 说明 | 示例 |
|---|--------|------|------|------|
| 1 | `app_id` | int | Steam 唯一应用 ID | `730` |
| 2 | `title` | str | 游戏标题 | `"Counter-Strike: Global Offensive"` |
| 3 | `date_release` | str/date | 发布日期 | `"2012-08-21"` |
| 4 | `rating` | float | Steam 评分（正面评价/总评价，范围 0-1） | `0.87` |
| 5 | `positive_ratio` | float | 正面评价占比（与 rating 可能冗余，需核实） | `0.85` |
| 6 | `user_score` | int | 评分对应的用户量级（如 1-10，具体含义待确认） | `8` |
| 7 | `price_original` | float | 原价（美元） | `14.99` |
| 8 | `price_final` | float | 折扣后价格（美元）；若与原价一致则表示未打折 | `14.99` |
| 9 | `discount` | float | 折扣比例（0-1）；0 = 无折扣 | `0.0` |
| 10 | `owners` | str | 拥有者估计区间 | `"1000000-2000000"` |
| 11 | `steam_deck` | int | Steam Deck 兼容性等级（0-2，具体编码待确认） | `1` |

> **注意：** `rating` 和 `positive_ratio` 的关系需要在实际数据中核实，可能存在功能重叠。`owners` 是范围文本（如 "1000000-2000000"），需要解析为数值。

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

> **注意：**
> - `hours` 是重要的补充特征：用户玩的时间越长，评价的可信度/权重可能越高
> - `helpful` 和 `funny` 反映社区对评价本身的认可度，可用于权重调整
> - `(user_id, app_id)` 理论上是唯一的，但可能存在历史修改记录（新旧评价并存），需按 `date` 去重保留最新

---

## games_metadata.json

游戏富文本元数据（非表格化 JSON，每行一个独立 JSON 对象）。

| # | 字段名 | 类型 | 说明 | 示例 |
|---|--------|------|------|------|
| 1 | `app_id` | int | Steam 唯一应用 ID（外键 → `games.app_id`） | `730` |
| 2 | `description` | str | 游戏描述文本（多语言混合） | `"The #1 competitive FPS..."` |
| 3 | `tags` | list[str] | 用户自定义标签数组 | `["FPS", "Multiplayer", "Shooter", "Competitive"]` |
| 4 | `genres` | list[str] | Steam 官方分类数组 | `["Action", "Free to Play"]` |
| 5 | `type` | str | 产品类型：`game` / `dlc` / `music` / `video` 等 | `"game"` |
| 6 | `early_access` | int | 是否为抢先体验（0/1） | `0` |

> **注意：**
> - `tags` 是用户自定义的，数量多、粒度细、有噪声（如 "Masterpiece"、"Walking Simulator"）
> - `genres` 是 Steam 官方分类，数量少、规范化程度高
> - `type` 可用于过滤——是否只保留 `game`，排除 `dlc`、`music`、`video`？
> - `description` 是多语言文本，初版分析可以不使用，后续可尝试 NLP 特征

---

## 字段使用优先级

按在分析中的必要性分三级：

### 核心字段（必须使用）

| 数据集 | 字段 | 用途 |
|--------|------|------|
| games | `app_id`, `title`, `rating`, `price_final`, `date_release` | 游戏基础画像 + 建模特征 |
| users | `user_id`, `products`, `reviews` | 用户行为画像 + 建模特征 |
| recommendations | `user_id`, `app_id`, `is_recommended`, `hours`, `date` | 交互数据 + 目标变量 |
| metadata | `app_id`, `tags`, `genres`, `type` | 类型特征 + 产品类型过滤 |

### 扩展字段（可用但非必需）

| 数据集 | 字段 | 用途 |
|--------|------|------|
| games | `discount`, `owners`, `steam_deck` | 可选的游戏侧特征 |
| recommendations | `helpful`, `funny` | 评价权重调整 |
| metadata | `early_access` | 是否为抢先体验（可能影响评分） |

### 待选字段（初版暂不使用）

| 数据集 | 字段 | 原因 |
|--------|------|------|
| games | `positive_ratio` | 可能与 `rating` 冗余 |
| games | `user_score` | 含义不明确 |
| metadata | `description` | 多语言文本，NLP 成本高 |
