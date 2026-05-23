# 04 — 发布日期趋势分析

## 描述
分析游戏发布量随时间的趋势，识别爆发期和低谷期。

## 依赖
- 数据清洗完成（`games_df` 含 `release_year` 列）

## 输入
- `games_df`（含 `release_year` 列）

## 输出
- 折线图（`outputs/figures/release_timeline.png`）+ 趋势解读

## 步骤
1. `df.groupby('release_year').size()` 统计每年发布量
2. 过滤异常年份（未来年份或 < 1990，Steam 不太可能在此前）
3. 画折线图，x 轴 = 年份，y 轴 = 发布数量
4. 计算 CAGR（复合年增长率）和逐年增长率
5. 标注关键事件年份：Steam Direct（2017）、Steam Greenlight 取消（2017）

## 验收标准
- 年份范围合理（约 2000-2025）
- 增长趋势和爆发期有明确解读
- 异常年份已被标记和解释
