# 06 — 超参数优化

## 描述
对最优候选模型（基于 ROC-AUC 初步结果）使用 `RandomizedSearchCV` 进行超参数调优。

## 依赖
- `03-logistic-regression`, `04-random-forest`, `05-xgboost` 中表现最好的模型

## 输入
- `X_train, y_train` + 选定的模型类

## 输出
- 最优超参数 + 最优模型 + CV 得分

## 步骤
1. 确定调优的模型（随机森林或 XGBoost）
2. 定义参数搜索空间（3-4 个关键参数）
3. `RandomizedSearchCV(estimator, param_distributions, n_iter=20, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42)`
4. `fit` 并打印 `best_params_` 和 `best_score_`
5. 用最优参数在完整训练集上重新训练

## 验收标准
- `best_score_` > 默认参数得分
- 最优参数在定义空间内部（非边界值）
- CV 标准差合理（< 0.02）
