# 05 — 合并 metadata 到 games

## 描述
将 `games_metadata.json` 展平后的 tags/genres 信息合并到 games 表。根据 `type` 字段决定是否过滤 DLC 等非游戏产品。

## 依赖
- `05-load-metadata`, `02-type-conversion`

## 输入
- `games_df`, `metadata_df`

## 输出
- 合并后的 `games_df`，新增 `tags`, `genres`, `type`, `early_access` 列

## 步骤
1. `games_df.merge(metadata_df, on='app_id', how='left')`
2. 打印 `type` 列的 `value_counts()`，了解产品类型分布
3. 对未匹配到的记录（`type` 为 NaN），打印数量和 `app_id`
4. 决定是否过滤：若 DLC/music/video 占比小且分析目标聚焦游戏，则 `df = df[df['type'] == 'game']`

## 验收标准
- 合并后游戏数量合理（匹配率 > 80%）
- `tags` 和 `genres` 列以 list 形式存在于 DataFrame 中
- 产品类型过滤决策已明确记录
