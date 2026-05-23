# 01 — 数据集划分

## 描述
按用户分组切分训练/测试集，确保同一用户的所有评价只在一个集合中，防止数据泄漏。

## 依赖
- `05-feature-engineering` 全部完成

## 输入
- `X`（特征矩阵）、`y`（目标变量）、`groups`（user_id）

## 输出
- `X_train, X_test, y_train, y_test`

## 步骤
1. 使用 `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)`
2. `train_idx, test_idx = next(gss.split(X, y, groups=user_id))`
3. 打印训练集/测试集的样本量、正例比例
4. 验证：`set(train_user_ids) & set(test_user_ids)` 应为空集

## 验收标准
- 训练集 : 测试集 ≈ 80% : 20%
- 训练集和测试集的 user_id 无交集
- 两集合的正例比例接近（分布一致）
