# 04 — 随机森林模型

## 描述
训练随机森林分类器。可捕获非线性关系，并提供特征重要性用于问题 4 分析。

## 依赖
- `01-data-split`

## 输入
- `X_train, y_train`

## 输出
- 训练好的 `RandomForestClassifier` 模型

## 步骤
1. `RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1, class_weight='balanced')`
2. 在训练集上 `fit`
3. 打印 Top 15 特征重要性
4. 在测试集上预测，计算 accuracy + roc_auc
5. 记录训练时间

## 验收标准
- ROC-AUC > 逻辑回归（预期）
- 特征重要性 Top 5 业务可解释
- 训练时间在合理范围（< 10 分钟，取决于数据量）
