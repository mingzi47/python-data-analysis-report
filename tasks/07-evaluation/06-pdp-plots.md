# 06 — 部分依赖图（PDP）

## 描述
绘制 Top 3-5 关键特征的部分依赖图，展示单个特征与推荐概率的边际关系。

## 依赖
- `05-feature-importance`（确定 Top 特征）

## 输入
- 最优模型 + `X_train`（或采样）+ Top 5 特征名

## 输出
- `outputs/figures/partial_dependence.png`（多子图并列）

## 步骤
1. 选取 Top 5 特征
2. `from sklearn.inspection import PartialDependenceDisplay`
3. 对每列调用 `PartialDependenceDisplay.from_estimator`
4. 或者手动：对每列取分位点，固定其他列为均值，画 predicted prob 线
5. 画 2×3 或 1×5 子图

## 验收标准
- 至少 3 个特征的 PDP 图
- 每条曲线的变化趋势有业务解读（例如：价格越高推荐率越低？）
- 曲线显示的是边际效应（非绝对值）
