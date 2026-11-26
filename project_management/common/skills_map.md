# 技能与工具边界索引（skills_map）

只登记本项目按需使用的正式 Skill 入口与研发/校验工具边界，不复制 Skill 正文。

来源仓库：`liujiaxinglab/KnowledgeBase`（具体使用时核对来源仓库最新正式版本）。

| 工具 / Skill | 来源路径 / 执行入口 | 适用场景与操作边界 |
| --- | --- | --- |
| 长期项目规划与执行 | `KnowledgeBase`：`project_content/20_skills/10_AI工具/05_长期项目规划与执行/SKILL.md` | 管理/规划 Agent 制定阶段 Subplan、拆合 Task 或做状态校准时使用；普通执行 Agent 不默认加载 |
| 通达信量化研究技能 | `KnowledgeBase`：`project_content/20_skills/30_TDX/01_量化研究技能/SKILL.md` | 核查 `TDX-01` 日线价格口径（未复权现货 vs 前复权）与指标计算习惯时按需查阅 |
| 通达信本地数据技能 | `KnowledgeBase`：`project_content/20_skills/30_TDX/02_本地数据技能/SKILL.md` | 排查 `QuantData/raw/tdx/vipdoc_parsed/` 批次命名与本地日线结构时按需查阅（本仓只读，不主动改写上游 `.day` 或 `QuantData`） |
| 自动化测试与静态检查 | `python -m pytest`（在已激活的 `quant` 环境中执行） | 运行单元/集成测试与跨目录导入回归；严禁硬编码个人解释器路径或擅自执行 `pip install` / `conda install` |
| 上游只读消费边界 | `QuantData` / `sector_radar` 只读文件 | 本仓仅只读读取 `TDX-01`、`BS-01` 交易日历、公司行为台账与 `sector_radar` 路由台账，严禁在本仓代码中联网抓数或修改外部仓库 |

本仓执行环境约束见：[本地量化Python执行环境约定.md](本地量化Python执行环境约定.md)。
