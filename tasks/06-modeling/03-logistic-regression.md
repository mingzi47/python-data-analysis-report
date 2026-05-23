# 03 — 逻辑回归模型

## 描述
训练逻辑回归作为线性分类基准。结果可解释，提供特征系数的方向和大小。

## 依赖
- `01-data-split`

## 输入
- `X_train, y_train`

## 输出
- 训练好的 `LogisticRegression` 模型

## 步骤
1. `LogisticRegression(max_iter=1000, random_state=42)`
2. 在训练集上 `fit`
3. 打印 Top 10 正系数特征（驱动推荐）和 Top 10 负系数特征（驱动不推荐）
4. 在测试集上预测，计算 accuracy + roc_auc 作为初步指标
5. 记录训练时间

## 验收标准
- 模型收敛（无 "failed to converge" 警告）
- 测试集 ROC-AUC 显著 > 0.5（至少 > 0.6）
- 特征系数的正负方向业务上合理
