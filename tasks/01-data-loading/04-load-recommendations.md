# 04 — 加载 recommendations.csv（采样）

## 描述
使用 pandas 加载 `recommendations.csv`。全量 4,100 万行过大，开发阶段用 `nrows` 采样 10-50 万条。

## 依赖
- `01-download-dataset`：需要数据集路径

## 输入
- `path + "/recommendations.csv"`

## 输出
- `pd.DataFrame`，包含采样后的推荐记录

## 步骤
1. 先用 `pd.read_csv(path + "/recommendations.csv", nrows=0)` 读列名，确认列
2. `pd.read_csv(path + "/recommendations.csv", nrows=500_000)` 采样 50 万条
3. 打印 `df.shape`、`df.info()`、`df.head(10)`
4. 打印 `df['is_recommended'].value_counts(normalize=True)` 查看推荐比例
5. 打印 `df['hours'].describe()` 查看游玩时长范围

## 验收标准
- `shape[0]` 约等于 `nrows`（除非数据总量 < nrows）
- `is_recommended` 仅有 0 和 1 两种值
- 采样量可配置（通过 `Config.sample_size` 控制）
