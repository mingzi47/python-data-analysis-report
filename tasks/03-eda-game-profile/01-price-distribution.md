# 01 — 价格分布分析

## 描述
分析游戏价格分布：免费 vs 付费占比、付费游戏的价格区间、是否存在价格断层。

## 依赖
- 数据清洗完成（`02-data-cleaning` 全部任务）

## 输入
- `games_df`（已清洗，含 `price_final` 列）

## 输出
- 统计摘要（打印）+ 直方图（路径：`outputs/figures/price_distribution.png`）

## 步骤
1. 统计免费游戏占比：`(df['price_final'] == 0).mean()`
2. 对付费游戏（price > 0）：`describe(percentiles=[.25, .5, .75, .9, .95, .99])`
3. 用 `plt.hist(price, bins=50)` 画直方图，x 轴做 log 变换
4. 用 `plt.pie([free, paid])` 画免费 vs 付费占比饼图
5. 标注关键分位数（中位数、$60 位置）

## 验收标准
- 免费游戏占比明确（预期约 20-40%）
- 直方图能看出价格分布形态
- 是否存在"价格断层"有明确判断
