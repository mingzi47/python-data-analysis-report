# 07 — 学习曲线

## 描述
绘制最优模型的学习曲线，判断是否欠拟合或过拟合。

## 依赖
- `02-model-comparison`（确定最优模型）

## 输入
- 最优模型类（未训练） + `X_train, y_train`

## 输出
- `outputs/figures/learning_curve.png`

## 步骤
1. `from sklearn.model_selection import learning_curve`
2. `train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)`
3. 计算 `train_mean = train_scores.mean(axis=1)` 和 `test_mean = test_scores.mean(axis=1)`
4. 画两条折线 + 误差带（shaded band）
5. x 轴 = 训练样本数，y 轴 = ROC-AUC

## 验收标准
- 训练和验证曲线均已绘制
- 能判断是否过拟合（gap 大小）、是否欠拟合（两条线都很低）
- 若过拟合不严重且验证曲线趋于平稳 → 模型可接受
