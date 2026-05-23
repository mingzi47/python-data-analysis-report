# 02 — 数据分析方法

本文档详细描述每个阶段的方法、工具、输入输出和具体操作步骤。

---

## 阶段 1 — 数据读取

**目标：** 将三个 CSV 文件和一个 JSON 文件加载到 Python 环境中，建立初步的数据认知。

### 读取方式

| 文件 | 读取方式 | 说明 |
|------|---------|------|
| `games.csv` | `pd.read_csv()` 全量 | 游戏数量有限（预计数万-十余万），可直接全量加载 |
| `users.csv` | `pd.read_csv()` 全量 | 用户数量较大但可一次性加载 |
| `recommendations.csv` | `pd.read_csv(nrows=N)` 采样 | 4,100 万行，开发阶段采样 10-50 万条 |
| `games_metadata.json` | `json.load()` 逐行解析后转 DataFrame | 每行是独立 JSON 对象，提取 `app_id` + `tags`（实际数据仅有 tags，无 genres/type/early_access） |

### 操作步骤

1. `kagglehub.dataset_download("antonkozyriev/game-recommendations-on-steam")` 获取缓存路径
2. `pd.read_csv()` 加载各文件，打印 `shape`、`info()`、`head()` 建立基本认知
3. 逐行读取 `games_metadata.json`，提取 `app_id`、`tags`，展平为 DataFrame（实际数据仅有这 3 个字段）
4. 检查主键唯一性：`games.app_id`、`users.user_id`、`recommendations.(user_id, app_id)` 组合

---

## 阶段 2 — 数据清洗与预处理

**目标：** 处理缺失值、异常值、数据类型转换、重复记录。

### 2.1 缺失值处理

| 检查项 | 方法 | 处理策略 |
|--------|------|---------|
| 缺失值统计 | `df.isnull().sum() / len(df)` | 先统计再决策 |
| 游戏标题缺失 | — | 删除该行（无法标识） |
| 游戏评分缺失 | — | 数值型中位数填充，文本型填充 "Unknown" |
| 发布日期缺失 | — | 填充数据集的中位数年份 |
| 用户行为数据缺失 | — | 填充为 0（无公开记录即无行为） |
| 推荐目标变量缺失 | — | 删除该行（目标变量不可推测） |

### 2.2 数据类型转换

- `date` → `pd.to_datetime()`，提取 `year`、`month`
- `price` → `float`，检查非数字字符（如 "Free"）
- `is_recommended` → `int`（0/1）
- `tags` → 从 JSON 数组转为 Python list，用 explode 展开统计或 `apply(len)` 计算标签数量

### 2.3 异常值检测

- **价格：** IQR 方法，结合业务判断（> $500 可能是特殊版本，标记但不删除）
- **用户活跃度：** 评论数 > 已购产品数 是逻辑异常，标记并考虑排除
- **评分：** Steam 文本标签（"Very Positive" 等）无范围限制；数值型评分（`positive_ratio`）为 0-100，检查超范围记录

### 2.4 重复处理

- `recommendations.csv`：同一 `(user_id, app_id)` 保留最新一条（按时间戳）
- `games.csv`：相同 `app_id` 去重

---

## 阶段 3 — 探索性数据分析（EDA）

**目标：** 回答研究问题 1（游戏生态画像）和问题 2（用户行为画像）。

### 3.1 单变量分析 — 游戏维度

| 分析项 | 统计方法 | 可视化方法 |
|--------|---------|-----------|
| 价格分布 | `describe()` + 分位数 + 偏度/峰度 | 直方图 + KDE（log 轴）；饼图展示免费 vs 付费占比 |
| 评分分布 | `value_counts()`（文本型）或 `describe()`（数值型） | 柱状图（文本型）或直方图+KDE（数值型）；竖线标注均值和中位数 |
| 类型频率 | `value_counts()`（使用 `tags`，因实际数据无 `genres`） | 横向条形图（Top 20 标签） |
| 发布时间趋势 | `groupby('year').size()` | 折线图 |

### 3.2 单变量分析 — 用户维度

| 分析项 | 统计方法 | 可视化方法 |
|--------|---------|-----------|
| 评论数分布 | `describe(percentiles=[.25, .5, .75, .9, .95, .99])` | 直方图（log-log 坐标） |
| 购买数分布 | 同上 | 直方图（log-log 坐标） |
| 推荐率分布 | 每位用户的 `is_recommended.mean()` | 直方图 + 标注 |

### 3.3 双变量与多变量分析

| 分析项 | 方法 | 可视化 |
|--------|------|--------|
| 价格 vs 评分 | Spearman 秩相关系数 | 散点图 + LOWESS 平滑 |
| 类型 vs 评分 | 按标签分组计算平均评分（使用 tags） | 分组柱状图 |
| 发布时间 vs 评分 | 按年份分组 | 折线图 |
| 用户活跃度 vs 推荐倾向 | 按活跃度分层（low/medium/high/extreme） | 箱线图 |
| 头部集中度 | Gini 系数 + Lorenz 曲线 | 长尾分布曲线（log-log） |

### 3.4 关键指标计算

- Gini 系数（游戏推荐量的不平等程度）
- 用户平均推荐率
- 付费 vs 免费游戏推荐率差异（Mann-Whitney U 检验）
- 各年份游戏发布量 CAGR（复合年增长率）

---

## 阶段 4 — 可视化（EDA 阶段）

**工具：** `matplotlib` + `seaborn`

**视觉规范：**
- 统一配色（seaborn `color_palette`）
- 中文/英文标签统一，字号 >= 12pt
- 每张图包含标题、轴标签、图例
- 输出格式：PNG（报告嵌入）+ SVG（可选，矢量）
- 保存路径：`outputs/figures/`

### 图表清单

| # | 图表 | 类型 |
|---|------|------|
| 1 | 游戏价格分布 | 直方图（log 轴） |
| 2 | 游戏评分分布 | 柱状图（文本型）或 直方图+KDE（数值型） |
| 3 | Top 20 游戏标签/类型 | 横向条形图（实际数据使用 tags） |
| 4 | 年度游戏发布量 | 折线图 |
| 5 | 用户评论数分布 | 直方图（log-log） |
| 6 | 用户推荐率分布 | 直方图 |
| 7 | 用户活跃度分布（产品/评论） | 双直方图（log 轴） |
| 8 | 用户推荐率分布 | 直方图 + 标注线 |
| 9 | 购买数 vs 评论数 | 散点图（log-log）+ Spearman 相关 |
| 10 | 用户活跃度分层 vs 推荐率 | 箱线图 |
| 11 | 游戏推荐量长尾分布 | log-log 曲线 |
| 12 | 数值变量相关性 | 热力图 |

### 分析辅助函数

以下指标在流水线中自动计算并输出：

| 指标 | 函数 | 输出 |
|------|------|------|
| 推荐量集中度 | `compute_concentration_metrics()` | Gini 系数 + Top1%/5%/10%/20% 占比 |
| 极端用户识别 | `analyze_extreme_users()` | 纯好评/纯差评用户占比及按评价数分桶 |
| 活跃度分层 | `compute_user_activity_tiers()` | low/medium/high/extreme 四层分类 |

---

## 阶段 5 — 特征工程

**目标：** 从原始字段构造特征矩阵 X 和目标向量 y。

### 5.1 目标变量

- `y = recommendations.is_recommended`（0/1 二分类）

### 5.2 特征构造

#### （A）游戏侧特征（来自 `games.csv` + `games_metadata.json`）

| 原始字段 | 处理方式 | 输出特征 |
|---------|---------|---------|
| `price` | 直接使用 + 二值化 | `price`, `is_free` |
| `rating` | 数值型直接使用；文本型回退为 `positive_ratio / 100` | `rating` (0-1 数值) |
| `date` | 提取年份，计算距今 | `release_year`, `years_since_release` |
| `tags` | `apply(len)` 计算数量；可用 expand 展开统计 | `num_tags` |
| `genres` | 实际数据无此字段，`num_genres` 回退为 `num_tags` | `num_genres` (= num_tags) |
| `early_access` | 实际数据无此字段，列存在时填充 0 | `early_access` (0/1) |

#### （B）用户侧特征（来自 `users.csv`）

| 原始字段 | 处理方式 | 输出特征 |
|---------|---------|---------|
| `products` | 直接使用 | `user_products_count` |
| `reviews` | 直接使用 | `user_reviews_count` |
| — | 衍生 | `review_ratio = reviews / products` |

#### （C）交互特征（从 `recommendations.csv` 聚合）

| 特征 | 构造方式 | 含义 |
|------|---------|------|
| `game_review_count` | 按 `app_id` 统计条数 | 游戏热度 |
| `game_recommend_rate` | 按 `app_id` 统计 `mean(is_recommended)` | 游戏整体口碑 |
| `user_recommend_rate` | 按 `user_id` 统计 `mean(is_recommended)` | 用户评价倾向 |
| `user_review_count` | 按 `user_id` 统计条数 | 用户在数据集的活跃度 |

> **注意：** 交互特征聚合必须在训练集上计算后映射到测试集，或用时间切分只用历史数据，防止数据泄漏。

### 5.3 特征处理流水线

使用 `ColumnTransformer` 构建可复用流水线：

```
ColumnTransformer:
├── 数值特征（price, rating, hours, user_products_count, game_recommend_rate, ...）
│   └── StandardScaler（当前默认 passthrough 全量特征）
├── 二值特征（is_free）
│   └── passthrough
└── 其余特征
    └── passthrough
```

### 5.4 特征选择（可选）

- `SelectKBest`（mutual information）或树模型特征重要性筛选
- `VarianceThreshold` 移除方差接近 0 的特征

---

## 阶段 6 — 建模与优化

**目标：** 训练并对比多个分类模型，选出最优模型并调参。

### 6.1 数据集划分

```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=recommendations['user_id']))
```

- 训练集 : 测试集 = 80% : 20%
- 组标识 = `user_id`，确保同一用户的所有评价只在一个集合中

### 6.2 基线模型

| 基线 | 方法 | 意义 |
|------|------|------|
| 随机猜测 | `DummyClassifier(strategy='uniform')` | 理论下限（50%） |
| 多数类预测 | `DummyClassifier(strategy='most_frequent')` | 总是预测"推荐"的准确率 |
| 简单规则 | `game_recommend_rate > 0.5` 则预测推荐 | 纯基于口碑的朴素规则 |

### 6.3 候选模型

| 模型 | 角色 | 关键参数 |
|------|------|---------|
| **逻辑回归** (`LogisticRegression`) | 线性基准，可解释性强 | `C`, `penalty`, `solver` |
| **随机森林** (`RandomForestClassifier`) | 非线性，特征重要性可解释 | `n_estimators`, `max_depth`, `min_samples_split` |
| **XGBoost** (`XGBClassifier`) | 梯度提升，通常性能最强 | `n_estimators`, `max_depth`, `learning_rate`, `subsample` |

### 6.4 超参数优化

- `RandomizedSearchCV`，`n_iter=20`，`cv=3`
- 评估指标：`scoring='roc_auc'`

### 6.5 训练注意事项

- XGBoost 启用 `early_stopping_rounds`
- 树模型设置 `class_weight='balanced'` 处理类别不平衡
- 记录每个模型的训练时间

---

## 阶段 7 — 模型评估与分析

### 7.1 性能指标

| 指标 | 含义 | 适用场景 |
|------|------|---------|
| Accuracy | 整体正确率 | 类别均衡时参考 |
| Precision | 预测"推荐"中真正推荐的比例 | 关心推荐质量 |
| Recall | 真正推荐中被正确识别的比例 | 关心覆盖 |
| F1 Score | Precision 和 Recall 的调和平均 | 两者需要平衡 |
| ROC-AUC | 排序能力 | 整体判别能力 |
| Confusion Matrix | TP/FP/FN/TN | 错误类型分布 |

### 7.2 对比呈现

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC | 训练时间 |
|------|----------|-----------|--------|-----|---------|---------|
| Dummy (uniform) | | | | | | |
| Dummy (most_frequent) | | | | | | |
| 简单规则 | | | | | | |
| 逻辑回归 | | | | | | |
| 随机森林 | | | | | | |
| XGBoost | | | | | | |

### 7.3 深入分析（最优模型）

- **ROC 曲线：** 各模型叠在一张图上对比
- **混淆矩阵：** 热力图
- **特征重要性：** Top 15 横向条形图
- **部分依赖图（PDP）：** Top 3-5 关键特征
- **学习曲线：** 性能随样本量的变化

---

## 阶段 8 — 结果可视化与应用结论

### 8.1 最终可视化清单

1. 模型 ROC 对比图（所有模型叠加 + AUC 标注）
2. 混淆矩阵热力图（最优模型）
3. 特征重要性 Top 15（横向条形图）
4. 部分依赖图（3-5 个子图并列）
5. 学习曲线（训练集和验证集性能）

### 8.2 结论框架

| 受众 | 关注问题 | 产出 |
|------|---------|------|
| 游戏开发者 | 什么因素影响推荐意愿？ | 定价策略和类型选择建议 |
| 平台运营 | 冷启动问题严重吗？ | 推荐系统改进建议 |
| 数据科学 | 模型性能上限在哪？ | 特征工程 vs 协同过滤对比，后续改进方向 |

### 8.3 局限与改进方向

- 数据仅限于 Steam 平台，结论不一定能外推到主机或移动端
- 用户特征仅有行为数据，缺少年龄/地区/语言等人口统计特征
- 推荐是显式反馈，存在自选择偏差（极端用户更倾向评价）
- 未使用协同过滤（SVD、NCF），可在后续工作中对比
