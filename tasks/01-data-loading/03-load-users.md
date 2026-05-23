# 03 — 加载 users.csv

## 描述
使用 pandas 加载 `users.csv`，全量读取，打印基本信息确认数据结构。

## 依赖
- `01-download-dataset`：需要数据集路径

## 输入
- `path + "/users.csv"`

## 输出
- `pd.DataFrame`，包含所有用户记录

## 步骤
1. `pd.read_csv(path + "/users.csv")` 全量加载
2. 打印 `df.shape`、`df.info()`、`df.head(10)`、`df.describe()`
3. 检查 `user_id` 的 `dtype`（应为 int 或 str）
4. 检查 `products` 和 `reviews` 的值域是否合理（>= 0）

## 验收标准
- `shape[0]` > 0
- `user_id` 唯一性待后续任务验证
- `products` 和 `reviews` 均为非负数值
