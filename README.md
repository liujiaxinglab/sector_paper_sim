# sector_paper_sim

## 1. 项目定位

`sector_paper_sim` 是承接上游 `sector_radar` 四态定档与 `QRI` / `QuantData` 权威数据底座的**交易日日频步进确定性模拟撮合与五账户基准对照引擎**。本项目面向「`159330.SZ`（沪深300ETF 基准）+ 6 只行业 ETF（`159622.SZ`、`510720.SH`、`512400.SH`、`512480.SH`、`512880.SH`、`515790.SH`）」，以可完全重放的日频确定性状态机并行追踪五组虚拟账户（`Account A` 四态路由实验盘、`Account B1` 纯机械海龟、`Account B2` 纯机械日频网格、`Account C` 行业等权持有、`Account D` 沪深300持有），在 12–18 个月实证运行期内持续检验信息路由相对纯机械规则与宽基基准的真实增量。

## 2. 多仓库协同边界

| 仓库 | 角色定位 | 与本仓访问边界 | 消费 / 产出摘要 |
| --- | --- | --- | --- |
| `KnowledgeBase` | 公共项目管理规范与经验库 | **只读引用**（严禁写入） | 提供 V2 项目管理协议（`00`–`03`）与共享数据复用规范 |
| `QRI`（`quant_research_infrastructure`） | 本地量化数据采集底座 | **只读对齐**（严禁写入） | 提供 `TDX-01` 日线与 `BS-01` 交易日历数据采集能力规范 |
| `QuantData` | 本地权威量化数据仓 | **只读消费**（严禁写入） | 提供 `TDX-01` ETF 日线、`BS-01` 交易日历及公司行为台账 |
| `sector_radar` | 行情定性与四态路由器 | **只读消费**（严禁写入） | 提供观察池配置、`regime_history.json` 四态路由历史与实盘对照基准 |
| `sector_paper_sim`（本仓） | 日频确定性模拟撮合引擎 | **独立维护与落盘** | 消费上述只读上游，产出结构化日频净值、成交明细与对账报告 |

## 3. 总体阶段路线

1. **[阶段 1｜初版引擎搭建（历史归档）](project_management/subplan/1%20初版引擎搭建（历史归档）/1%20子规划.md)**：完成初版建仓与周频原型试跑；`2026-11-26` 经 V2 审计发现 9 项协议、撮合规则与可复现性问题（`ISS-001`–`ISS-009`）后作废冻结归档。
2. **[阶段 2｜日频化与合规重构](project_management/subplan/2%20日频化与合规重构/2%20子规划.md)**（已完成，`2026-11-27` 主控终审通过）：`2t01`–`2t04` 全部收口通过，真实上游重锚、全量/增量一致性与双仓对账证据已归档。
3. **阶段 3｜日频常态化跟随对账运行（已准入，自 `2026-11-27` 起）**：阶段 2 主控终审与 Radar 阶段 5 联合验收已于 `2026-11-27` 通过，`forward_eval_start_date` 已定为 `2026-11-27`；待补齐目标交易日上游数据后随交易日步进并对账检验。

## 4. 项目管理与业务入口

- 统一项目管理协议（`liujiaxinglab/KnowledgeBase`）：`project_content/10_常用规则/03_项目管理协议.md`
- 当前状态：[project_management/status.md](project_management/status.md)
- 路线决策记录：[project_management/路线决策记录.md](project_management/路线决策记录.md)
- 问题追踪台账：[project_management/issues.md](project_management/issues.md)
- 规则与技能索引：[rules_map.md](project_management/common/rules_map.md) ｜ [skills_map.md](project_management/common/skills_map.md) ｜ [本地量化Python执行环境约定.md](project_management/common/本地量化Python执行环境约定.md)
- 撮合与切换确定性规则：[project_content/docs/01_撮合与切换确定性规则.md](project_content/docs/01_撮合与切换确定性规则.md)
- 实验复现与真实事件抽核：[project_content/docs/03_实验复现与真实事件抽核.md](project_content/docs/03_实验复现与真实事件抽核.md)
- 唯一模拟参数配置：[project_content/config/sim_config.json](project_content/config/sim_config.json)
