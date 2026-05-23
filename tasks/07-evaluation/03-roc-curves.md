# 03 — ROC 曲线对比图

## 描述
在一张图上叠加所有模型的 ROC 曲线，标注 AUC 值。

## 依赖
- `01-metrics-calculation`（需要 `y_proba`）

## 输入
- 各模型的 `y_proba` + `y_test`

## 输出
- `outputs/figures/roc_curves.png`

## 步骤
1. 对每个有 `predict_proba` 的模型，计算 `fpr, tpr, _ = roc_curve(y_test, y_proba)`
2. 对 Dummy 模型，画对角线（(0,0) → (1,1)）
3. 不同模型用不同颜色，图例标注 `Model (AUC=0.xxx)`
4. 添加 x=y 参考线

## 验收标准
- 至少 4 条 ROC 曲线（逻辑回归、随机森林、XGBoost、XGBoost-Tuned）
- AUC 值在图例中标注
- 视觉效果清晰可读
