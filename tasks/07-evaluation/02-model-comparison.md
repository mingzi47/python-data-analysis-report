# 02 — 模型对比表格

## 描述
将各模型性能指标汇总为对比表格，打印并保存。

## 依赖
- `01-metrics-calculation`

## 输入
- 指标字典

## 输出
- 格式化对比表格（Markdown / DataFrame 打印）+ `outputs/models/comparison.csv`

## 步骤
1. 将指标字典转为 `pd.DataFrame`，行 = 模型，列 = 指标
2. 按 `roc_auc` 降序排列
3. 打印格式化表格
4. 保存为 CSV：`outputs/models/comparison.csv`
5. 标注最优模型的每个指标

## 验收标准
- 表格包含所有 7 个模型
- 最优模型在多个指标上表现最好
- CSV 已保存
