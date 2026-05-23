# 02 — 用户推荐偏好分析

## 描述
分析用户整体推荐率分布，以及不同活跃度分层的推荐倾向差异。

## 依赖
- `01-activity-distribution`：需要活跃度分层标准

## 输入
- `recommendations_df` + `users_df`（已分层）

## 输出
- 直方图（`outputs/figures/user_recommend_rate.png`）+ 分层对比

## 步骤
1. 计算每位用户的推荐率：`user_rate = df.groupby('user_id')['is_recommended'].mean()`
2. 画 `user_rate` 的直方图，标注 0.5 线和总体均值线
3. 按活跃度分层统计各层用户的平均推荐率
4. 用箱线图展示不同活跃度分层的推荐率分布
5. 做统计检验：活跃度最高层 vs 最低层的推荐率是否有显著差异

## 验收标准
- 用户推荐率分布形态清晰
- 不同活跃度层的推荐倾向差异有统计检验结论
- 能回答"活跃用户更严格还是更宽松"
