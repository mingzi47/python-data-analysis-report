# 04 — 混淆矩阵热力图

## 描述
对最优模型画混淆矩阵热力图，分析错误类型分布。

## 依赖
- `02-model-comparison`（确定最优模型）

## 输入
- 最优模型的 `y_pred` + `y_test`

## 输出
- `outputs/figures/confusion_matrix.png`

## 步骤
1. `confusion_matrix(y_test, y_pred)` 计算四格表
2. 归一化为百分比（每行和为 1）
3. 用 `sns.heatmap` 画热力图，标注数值
4. 轴标签：实际值（Actual）/ 预测值（Predicted）
5. 在标题中标注模型名和 Accuracy

## 验收标准
- TP, FP, FN, TN 四格清晰
- 归一化百分比已标注
- False Positive 和 False Negative 的比例有业务解读
