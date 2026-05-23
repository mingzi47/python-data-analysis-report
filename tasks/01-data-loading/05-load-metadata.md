# 05 — 加载并解析 games_metadata.json

## 描述
`games_metadata.json` 是非表格化数据，每行一个独立 JSON 对象。逐行读取并提取 `app_id`、`tags`、`genres`、`type`、`early_access` 字段，展平为 DataFrame。

## 依赖
- `01-download-dataset`：需要数据集路径

## 输入
- `path + "/games_metadata.json"`

## 输出
- `pd.DataFrame`，列：`app_id`, `tags` (Python list), `genres` (Python list), `type`, `early_access`

## 步骤
1. `with open(path + "/games_metadata.json") as f:` 逐行读取
2. 每行 `json.loads(line)` 解析为 dict
3. 提取 `app_id`, `tags`, `genres`, `type`, `early_access`，缺失字段填 `None` 或 `[]`
4. 将提取的 dict 列表转为 DataFrame
5. 打印 `df.shape`、`df['type'].value_counts()` 了解产品类型分布

## 验收标准
- `shape[0]` > 0 且与 `games.csv` 行数可比
- `tags` 和 `genres` 列为 list 类型
- `type` 列包含 `"game"`, `"dlc"` 等值
