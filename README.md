# 接单时的外卖晚到风险识别

本项目是 **MIS 7005 AI and Big Data Intelligence** 的小组课程研究。我们使用美团公开发布的配送数据，研究一个具体的运营问题：**在骑手接单时，如果运营人员只能关注少量订单，模型能否优先识别随后晚于预计送达时间的订单？**

项目比较随机排序、距离规则、Logistic Regression 和 XGBoost，并将排序表现转化为监控工作量、预警精确率和晚到覆盖率。完整分析位于 [food_delivery_analysis.ipynb](notebooks/food_delivery_analysis.ipynb)，包括数据处理、模型训练、时间顺序验证、误差分析和结果解释。

当前成果是历史数据上的预测分析与可复现 Notebook。项目已导出供 Dashboard 使用的指标和逐单评分，但尚未包含可交互的 Dashboard 应用，也未验证实际干预能够减少多少晚到。

## 1. 数据来源与使用范围

### 官方来源

数据来自美团为 **INFORMS TSL Data-Driven Research Challenge** 发布的运营配送数据集。官方研究挑战允许研究者使用这些数据研究配送运营问题，也允许探索自行提出的研究问题。来源和获取入口如下：

- [美团官方数据仓库](https://github.com/meituan/Meituan-INFORMS-TSL-Research-Challenge)：原始数据、许可文件及字段说明。
- [INFORMS TSL 研究挑战介绍](https://connect.informs.org/tsl/tslresources/datachallenge)：数据发布背景与研究范围。
- [数据说明论文的本地副本](<data/Meituan-INFORMS-TSL-Research-Challenge/Wang-etal-2025_Meituan_Operational-Level On-Demand Food Delivery Data for the 2025 INFORMS TSL Data-Driven Research Challenge.pdf>)：Wang 等人的 *Meituan: Operational-Level On-Demand Food Delivery Data for the 2025 INFORMS TSL Data-Driven Research Challenge*，说明数据结构、事件时间和坐标处理方法。

本项目实际读取运单主表 `all_waybill_info_meituan_0322.csv`，原始表包含 **654,343 条运单记录**。数据说明中的业务窗口为 **2022 年 10 月 17–24 日**；研究挑战的年份不代表订单发生年份，个别记录的下单时间也可能早于该窗口。一个订单可以对应多次派单，因此运单行数不能直接视为独立订单数。

### 与课程要求的对应关系

课程指南允许采用公开数据、爬虫数据或获授权的企业数据，要求围绕真实商业问题开展数据处理、AI/机器学习分析、可视化和管理讨论。本项目选择提供方正式公开发布的数据，研究有限监控资源下的配送晚到识别问题，符合课程关于数据来源类型与业务分析主题的要求。

| 课程要求 | 本项目的具体做法 |
|---|---|
| 明确具有商业意义的问题 | 在骑手接单时识别晚到风险，评估有限监控容量下的工作量与识别效果 |
| 说明数据来源、清洗与结构化过程 | 保留官方来源与许可，记录样本筛选、时间处理、标签定义和特征构造 |
| 使用 AI/机器学习方法并评价 | 训练 Logistic Regression 与 XGBoost，与随机及距离基准比较 |
| 提供可视化、定量分析和管理含义 | 展示容量曲线、SHAP、分组错误率及 ETA 提前量，并说明运营解释的边界 |
| 讨论数据伦理、隐私与可解释性 | 说明使用条件、位置处理、误报与漏报负担，区分模型解释与因果结论 |
| 提交可检查的代码或 Notebook | 提供完整 Notebook、依赖版本、运行步骤和汇总结果 |

上述对应关系说明当前分析如何落实课程要求。最终报告、演示及其提交仍需按课程安排完成；Notebook 中的实验结果不能代替完整的报告和伦理隐私讨论。

### 数据许可与非商业用途

本项目将数据用于非商业的课程学习与学术分析，这一用途与提供方列明的教学、学术研究用途一致。数据的具体使用条件以随附的 [License.txt](data/Meituan-INFORMS-TSL-Research-Challenge/License.txt) 及[官方许可文件](https://github.com/meituan/Meituan-INFORMS-TSL-Research-Challenge/blob/main/License.txt)为准。

许可文件标明 **CC BY-NC 4.0**，同时明确写明：未经许可不得向第三方分发数据集，数据及其衍生作品不得用于商业目的，公开发表相关研究时应按要求致谢。因此，非商业研究用途符合许可列明的用途范围，并不等于已经获得重新分发原始数据的许可。

原始 CSV 与压缩包保留在本地，由 `.gitignore` 排除；获取项目代码后，应从提供方的官方渠道取得数据。逐单评分文件也作为可再生成的本地输出排除，版本控制保留数据字典、许可、来源说明和汇总指标。

## 2. 项目结构

```text
ai-bigdata-project/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── data_dictionary.csv
│   └── Meituan-INFORMS-TSL-Research-Challenge/
│       ├── License.txt
│       ├── *.pdf                              # 官方背景及数据说明
│       ├── all_waybill_info_meituan_0322.csv   # 本地输入，Git 忽略
│       ├── all_waybill_info_meituan_0322.csv.zip
│       └── 其他原始 CSV                       # 本地保留，当前分析未使用
├── notebooks/
│   ├── food_delivery_analysis.ipynb
│   └── Presentation_Handoff.md               # 演示与最终报告的交接说明
└── results/
    ├── dashboard_metrics.csv                  # 整体及逐日指标
    └── dashboard_order_scores.csv             # 逐单评分，Git 忽略
```

[数据字典](data/data_dictionary.csv)是本项目对原始字段含义和分析用途的整理，字段解释应结合官方说明阅读。当前 Notebook 只依赖运单主表和数据字典；`dispatch_rider_meituan.csv`、`dispatch_waybill_meituan.csv` 与 `courier_wave_info_meituan.csv` 不参与本轮训练。

`plan/`、`review/`、`project-requirement/` 及 `.omx/` 属于本地规划、审阅、课程参考或运行材料，已从版本控制中排除。[演示交接文件](notebooks/Presentation_Handoff.md)继续保留，供报告和演示制作人员接续工作。忽略规则不会删除本地文件，也不会自动移除已经被 Git 跟踪的文件。

## 3. 研究设计

### 预测时点与标签

预测时点为骑手接单时间 `grab_time`。预计送达时间简称 ETA，对应数据中的 `estimate_arrived_time`。

| 标签 | 正例定义 | 用途 |
|---|---|---|
| `y_0` | 实际送达时间严格晚于记录 ETA | 主分析，识别所有超过 ETA 的订单 |
| `y_8` | 实际送达时间比记录 ETA 晚超过 8 分钟 | 敏感性分析，检验对较明显延迟的识别能力 |

两个标签的正例比例不同。模型与基准的精确率、覆盖率在同一标签内比较，不通过跨标签的指标高低判断哪个模型更好。

### 样本清洗与时间划分

清洗保留已接单，且接单、实际送达与 ETA 时间有效的记录。筛选后共有 **568,545 笔订单**，订单 ID 无重复。非正时间戳转为缺失，负阶段耗时不参与统计；极端正耗时经分位数和记录核查后保留，因为现有证据不足以将其统一认定为错误。

全部日期和小时采用北京时间（`Asia/Shanghai`）。数据按接单日期划分，并限制训练、验证阶段的标签可得时间：

| 数据集 | 接单日期 | 标签截止规则 | 最终订单数 |
|---|---|---|---:|
| 训练集 | 2022-10-17 至 2022-10-21 | 必须在 10 月 22 日零点前送达 | 345,355 |
| 验证集 | 2022-10-22 | 必须在 10 月 23 日零点前送达 | 77,230 |
| 测试集 | 2022-10-23 至 2022-10-24 | 使用事后可得的有效标签评估 | 145,068 |

训练和验证窗口分别有 423、407 笔订单因标签截止规则被排除，另有 62 笔清洗后订单的接单日期不在指定窗口内。三个最终数据集的订单 ID 不重叠。

### 特征与比较方法

模型使用商家至顾客距离、骑手接单位置至商家距离、派单至接单耗时、进入调度至接单耗时，以及接单小时、下单小时、预约类型和运营区域，并增加数值缺失标记。距离由经过统一偏移的坐标估算，表示近似地理距离，不代表实际道路行驶距离。

填补、标准化和类别编码仅在训练集拟合，验证与测试使用同一转换器。取餐、实际送达及接单后的阶段耗时不进入预测特征。记录 ETA 仅用于标签和回溯提前量分析，其历史版本是否与接单时可见版本一致仍未确认。

主分析比较四种方法：固定种子的随机排序、商家至顾客距离规则、Logistic Regression 和 XGBoost。`y_8` 使用相同特征与 XGBoost 超参数单独训练，并与同标签的随机和距离基准比较。项目未进行大规模超参数搜索。

### 从模型分数到监控工作量

验证集先按分数排序，比较监控前 5%、10%、20% 订单时的表现；随后选择使监控量接近目标容量的分数阈值，并将阈值固定应用于测试集。同分订单整体纳入阈值判断。

**目标容量与实际监控率是不同的量。** 例如，10% 是验证集上的目标；测试集的分数分布变化后，同一阈值对应的工作量可以高于或低于 10%。测试集不重新选择阈值来强行满足目标比例。

| 指标 | 定义与含义 |
|---|---|
| 实际监控率 | 预警订单数 / 全部订单数，表示监控工作量 |
| 精确率（Precision） | 命中订单数 / 预警订单数，表示预警中的实际晚到比例 |
| 覆盖率（Recall） | 命中订单数 / 实际晚到订单数，表示找回了多少晚到订单 |
| 误报率（FPR） | 误报数 / 实际准时订单数，与预警中的误报比例不同 |
| 漏报率（FNR） | 漏报数 / 实际晚到订单数，等于 1 减去覆盖率 |
| AP 与 ROC-AUC | 衡量排序表现，不等同于某一固定容量下的精确率 |
| Brier 分数 | 预测分数与真实标签的均方误差，越低越好 |

## 4. 当前主要结果

下表来自 [dashboard_metrics.csv](results/dashboard_metrics.csv) 的 `overall` 行，适用于 **`y_0`、XGBoost、10% 验证目标容量对应的固定阈值**，测试日期为 2022 年 10 月 23–24 日。

| 项目 | 结果 |
|---|---:|
| 测试订单数 | 145,068 |
| 实际晚到订单数 | 21,922 |
| 固定阈值 | 约 0.256661 |
| 触发预警 | 15,117 笔 |
| 实际监控率 | 10.42% |
| 命中晚到订单 | 4,537 笔 |
| 精确率 | 30.01% |
| 覆盖率 | 20.70% |
| 误报率 / 漏报率 | 8.59% / 79.30% |
| AP / ROC-AUC | 0.26763 / 0.67702 |
| Brier 分数 | 0.12137 |

这一结果表示，模型在监控约十分之一订单时识别了约五分之一的晚到订单，但仍漏掉了大部分晚到订单。表中阈值仅为方便阅读而取近似值；Notebook 和导出文件保留完整精度，不能用显示值替代实际判断阈值。

业务上，命中晚到并不一定意味着来得及提前处理。以 `lead_min = (记录 ETA − 接单时间) / 60` 计算，4,537 笔命中中有 **4,461 笔仍有正提前量**，**76 笔已到或超过记录 ETA**。其中 **4,237 笔至少提前 15 分钟**，占正提前量命中的 94.98%。这些数值描述相对于记录 ETA 的潜在处置时间，不能直接解释为可避免的晚到数量。

## 5. 环境与运行步骤

### 准备 Python 环境

本项目已在 **Python 3.13.15** 环境中验证。主要依赖版本记录在 [requirements.txt](requirements.txt)：pandas 3.1.0、NumPy 2.5.4、scikit-learn 1.9.1、XGBoost 3.4.1、Matplotlib 3.11.2，以及 JupyterLab 等运行依赖。其他版本组合尚未验证。

以下命令适用于 macOS / Linux，在项目根目录执行；`python3.13` 需指向已安装的 Python 3.13 环境。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows 可使用已安装的 Python 3.13 创建 `.venv`，并通过 `.\.venv\Scripts\Activate.ps1` 激活；后续 `python -m ...` 命令相同。SHAP 贡献使用 XGBoost 原生接口计算，无需另装 `shap` 包。

### 获取数据

1. 阅读[官方数据仓库](https://github.com/meituan/Meituan-INFORMS-TSL-Research-Challenge)中的许可和数据说明。
2. 从该仓库获取 `all_waybill_info_meituan_0322.csv.zip`，放入 `data/Meituan-INFORMS-TSL-Research-Challenge/`。
3. 解压后确认该目录直接包含 `all_waybill_info_meituan_0322.csv`，不要额外嵌套一层文件夹。

运行前必须具备这两个文件：

```text
data/Meituan-INFORMS-TSL-Research-Challenge/all_waybill_info_meituan_0322.csv
data/data_dictionary.csv
```

原始 CSV 不随代码版本提供。其余三份原始数据表不是当前 Notebook 的运行前提。

### 执行 Notebook

在已激活的环境中，从项目根目录继续执行：

```bash
cd notebooks
python -m jupyterlab food_delivery_analysis.ipynb
```

选择该环境对应的 Python 内核，重启内核后从头依次运行全部单元。**内核工作目录必须为 `notebooks/`**：代码通过 `../data/` 读取输入，并通过 `../results/` 写出结果。直接在项目根目录作为内核工作目录运行，会使这些相对路径指向错误位置。

Notebook 会训练模型并自动创建结果目录，最后覆盖同名的评分和指标 CSV。需要保留旧版本结果时，应在重新运行前另存。随机种子固定为 42；替换原始数据或依赖版本后，应根据实际输出同步更新文字中的数值。

### 常见运行问题

| 现象 | 检查方法 |
|---|---|
| 提示找不到输入 CSV | 确认压缩包已解压、文件名准确，且内核工作目录为 `notebooks/` |
| `ModuleNotFoundError` | 确认安装依赖的 Python 与 Notebook 选用的内核属于同一环境 |
| 后面的单元提示变量未定义 | 重启内核并从头运行，避免只执行中间单元 |
| 重新运行后的阈值或指标不同 | 核对数据版本、依赖版本和完整运行顺序，并以本次输出为准 |

## 6. 输出与复现核对

| 文件 | 内容 | 版本控制 |
|---|---|---|
| `results/dashboard_metrics.csv` | 整体及逐日指标；当前为 1 行整体、2 行逐日汇总 | 保留 |
| `results/dashboard_order_scores.csv` | 每笔测试订单的风险分数、预警、真实标签、分组及提前量；当前 145,068 行 | 本地生成，忽略 |
| `notebooks/food_delivery_analysis.ipynb` | 方法、图表、完整输出与分析文本 | 保留 |

逐单文件每笔订单一行，`risk_score` 保留原始分数精度，`alert` 表示是否达到固定阈值，`outcome` 为 TP、FP、FN 或 TN。真实标签、混淆矩阵类别和 ETA 提前量属于历史评估信息，不能全部视为实时业务系统的已知输入。

Notebook 末尾会读回两个 CSV，核对订单 ID 唯一性、逐单预警是否与完整精度阈值一致，以及预警数、命中数和 Brier 分数是否与前文一致。2026 年 11 月 18 日的本地复核已顺序执行全部 **59 个代码单元**，运行中的断言通过；在隔离目录重新生成的两份 CSV 与既有结果逐字节一致。这是对当前数据和环境的复核记录，不代表其他平台或版本已验证。

本地 `model_comparison.csv` 和 `capacity_comparison.csv` 目前为空占位文件，已加入忽略规则。模型比较与容量比较的有效结果在 Notebook 对应表格中。

## 7. 解释边界与伦理隐私

数据只覆盖连续八天，且仅有一天验证、两天测试。时间顺序切分减少了时间信息混用，但不能证明模型在其他月份、节假日或不同运营环境下同样有效。标签截止规则排除了部分跨日未完成订单，结论适用于具有有效时间记录的已接单样本。

字段定义支持按接单时点构造特征，但在线读取时延、历史 ETA 版本和极端正耗时的业务有效性仍未确认。模型解释和分组误差反映历史样本中的关联：SHAP 贡献不是因果效应，分组 FPR / FNR 差异也不能单独证明算法公平或不公平。

数据说明论文记载位置坐标经过统一偏移，项目使用匿名标识关联记录。这些处理不等于已验证不存在重识别风险。Notebook 保留的输出中包含少量逐单预览和案例；`.gitignore` 只能排除匹配文件，不能清除 Notebook 中已经保存的这些内容。对外发布前，应核查保留输出的使用范围，并优先展示汇总统计和不含逐单标识的图表。

风险预警应辅助人工运营判断，不应作为处罚骑手或评价个人表现的唯一依据。当前研究尚未测量预警处理成本、干预效果或长期错误负担，不能声称模型已经降低了晚到率。

## 致谢

本研究由美团提供数据支持。

This research was supported by data provided by Meituan.
