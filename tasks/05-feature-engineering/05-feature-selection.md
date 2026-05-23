# 05 — 特征选择（可选）

## 描述
如果特征数量过多（> 50 列），使用方差过滤和互信息进行筛选。

## 依赖
- `04-preprocessor-pipeline`

## 输入
- 预处理后的特征矩阵 `X_scaled`

## 输出
- 筛选后的特征列名列表

## 步骤
1. `VarianceThreshold(threshold=0.01)` 移除几乎无变化的特征
2. `SelectKBest(mutual_info_classif, k=min(50, X.shape[1]))` 选 Top K 个最有信息量的特征
3. 打印被移除的特征名和原因
4. 打印保留的特征 Top 10（按互信息得分）

## 验收标准
- 筛选后特征数 <= 50
- 被移除的特征有记录
- 保留特征有互信息得分排序
