# 01 — 计算全部性能指标

## 描述
对每个模型（3 基线 + 3 模型 + 调优后模型）计算完整的性能指标集。

## 依赖
- `06-modeling` 全部完成

## 输入
- 各模型对象 + `X_test, y_test`

## 输出
- 指标字典 `{model_name: {accuracy, precision, recall, f1, roc_auc, train_time}}`

## 步骤
1. 对每个模型调用 `model.predict(X_test)` 得到 `y_pred`
2. 若模型有 `predict_proba`，调用 `predict_proba(X_test)[:, 1]` 得到 `y_proba`
3. 计算：`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`
4. 汇总为嵌套字典

## 验收标准
- 7 个模型的指标均已计算
- `roc_auc_score` 对 Dummy 模型可能为 0.5（正常）
- 所有值在 [0, 1] 范围内（除训练时间）
