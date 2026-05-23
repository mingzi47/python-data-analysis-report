# 02 — 加载 games.csv

## 描述
使用 pandas 加载 `games.csv`，全量读取（游戏数量有限），打印基本信息确认数据结构。

## 依赖
- `01-download-dataset`：需要数据集路径

## 输入
- `path + "/games.csv"`

## 输出
- `pd.DataFrame`，包含所有游戏记录

## 步骤
1. `pd.read_csv(path + "/games.csv")` 全量加载
2. 打印 `df.shape`（行数、列数）
3. 打印 `df.info()`（列名、类型、非空数量）
4. 打印 `df.head(10)` 检查前几行
5. 打印 `df.describe()` 数值列初步统计

## 验收标准
- `shape[0]` > 0
- 列名与 [数据字典](../../docs/04-data-dictionary.md#gamescsv) 一致
- 无明显的读取错误（如列未对齐、编码乱码）
