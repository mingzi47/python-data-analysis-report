# 05 — XGBoost 模型

## 描述
训练 XGBoost 分类器。通常性能最强的表格数据模型，支持早停。

## 依赖
- `01-data-split`

## 输入
- `X_train, y_train`

## 输出
- 训练好的 `XGBClassifier` 模型

## 步骤
1. `XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8, random_state=42, early_stopping_rounds=20)`
2. 从训练集中分出 10% 作为验证集用于早停：`eval_set=[(X_val, y_val)]`
3. 在完整训练集上 `fit`（启用早停）
4. 打印 Top 15 特征重要性（gain-based）
5. 在测试集上预测，计算 accuracy + roc_auc
6. 记录训练时间和最佳迭代轮数

## 验收标准
- ROC-AUC 与随机森林可比或更高
- 早停生效（`best_iteration < n_estimators`）
- 特征重要性 Top 5 与随机森林有重叠（交叉验证）
