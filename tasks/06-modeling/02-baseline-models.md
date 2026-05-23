# 02 — 基线模型

## 描述
训练并评估三个基线模型：随机猜测、多数类预测、基于游戏口碑的简单规则。

## 依赖
- `01-data-split`

## 输入
- `X_train, y_train, X_test, y_test`

## 输出
- 三个基线的性能指标字典

## 步骤
1. `DummyClassifier(strategy='uniform')` — 等概率随机猜
2. `DummyClassifier(strategy='most_frequent')` — 总是预测多数类
3. 简单规则：若 `game_recommend_rate > 0.5` 预测 1，否则 0（用训练集的 `game_recommend_rate` 做阈值）
4. 每个基线计算：accuracy, precision, recall, f1, roc_auc
5. 打印基线性能表

## 验收标准
- uniform 的 accuracy ≈ 50%
- most_frequent 的 accuracy = 多数类占比
- 简单规则至少有 1-2 个指标 > most_frequent
