# 05 — 特征重要性分析

## 描述
从树模型（随机森林/XGBoost）提取特征重要性，生成 Top 15 条形图。

## 依赖
- `02-model-comparison`（确定最优树模型）

## 输入
- 随机森林模型 或 XGBoost 模型 + 特征名列表

## 输出
- `outputs/figures/feature_importance.png` + 排名表

## 步骤
1. 从 `model.feature_importances_` 提取重要性
2. 与特征名配对，按重要性降序排列
3. 取 Top 15，画横向条形图（`sns.barplot`）
4. 区分用户侧特征和游戏侧特征（用颜色或标注）
5. 总结：用户侧 vs 游戏侧谁更重要？

## 验收标准
- Top 15 特征的重要性得分可视化
- 用户侧/游戏侧特征可区分
- 重要性排名业务可解释（回答研究问题 4）
