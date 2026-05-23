# 01 — 下载数据集

## 描述
使用 `kagglehub` 下载 `antonkozyriev/game-recommendations-on-steam` 数据集，返回本地缓存路径。

## 依赖
- 无前置任务

## 输入
- 无（自动从 Kaggle 下载）

## 输出
- 数据集本地路径 `path`（字符串）

## 步骤
1. `import kagglehub`
2. `path = kagglehub.dataset_download("antonkozyriev/game-recommendations-on-steam")`
3. `print(path)` 确认下载成功
4. `os.listdir(path)` 确认包含 `games.csv`, `users.csv`, `recommendations.csv`, `games_metadata.json`

## 验收标准
- 代码运行无报错
- 四个文件均存在于返回的路径中
