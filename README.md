# sector_paper_sim

## 1. 项目定位

`sector_paper_sim` 是独立、轻量的行业 ETF 确定性模拟盘仓库。本项目承接 `sector_radar` 的四态路由结论，面向「1 宽基基准（`159330.SZ`）+ 6 行业 ETF（`512400.SH`、`510720.SH`、`159622.SZ`、`515790.SH`、`512480.SH`、`512880.SH`）」构建可完全复现的日频模拟撮合引擎与周度净值账本，用于在 12–18 个月例行运行期内，客观检验信息路由相对沪深 300 基准与纯机械规则的真实增量。

核心设计原则：
- **单一事实配置**：全仓参数统一收口于标准 JSON 文件 [sim_config.json](project_content/config/sim_config.json)（禁止使用 YAML）。
- **确定性可复现**：每日严格按「除息再投 -> 开盘路由切换/`TRAP_OFF`强平 -> 盘中海龟/网格触价撮合 -> 收盘净值核算」4 步顺序执行，无随机性、无未来函数。
- **长期低维护**：完成阶段 1 引擎搭建后即退出开发 Task 流，进入 12–18 个月周五例行运行期，产出落盘于 `output/` 目录并纳入版本追踪。

## 2. 上游只读边界与物理隔离

本仓库与外部上游仓库严格执行**单向只读边界**，所有文件创建与写入操作仅限于 `D:/VibeCoding/Quant/sector_paper_sim/` 目录内，严禁修改任何上游或兄弟仓库文件：

| 仓库 / 数据目录 | 路径 | 访问权限 | 消费内容 |
| --- | --- | --- | --- |
| `QuantData`（TDX 日线） | `D:/VibeCoding/Quant/QuantData/raw/tdx/vipdoc_parsed` | **只读** | 7 只 ETF 本地 `TDX-01` 解析后的最新批次日线 CSV |
| `QuantData`（分红台账） | `D:/VibeCoding/Quant/QuantData/reference/etf/corporate_actions/20261120/watchlist_dividend_events.json` | **只读** | 官方核验 ETF 现金分红除息与每份分红事件 |
| `sector_radar` | `D:/VibeCoding/Quant/sector_radar/project_content/config/watchlist.json` | **只读** | 观察池标的元数据、四态路由 `current_regime` 与份额拆分 `split_events` |
| `KnowledgeBase` | `D:/VibeCoding/KnowledgeBase` | **只读** | V2 项目管理协议与方法论参考，严禁写入 |
| `youce_quant_510720_strategy` | 独立策略研究仓 | **隔离 / 禁写** | 独立仓位，无写入耦合 |

## 3. 目录结构

```text
sector_paper_sim/
├── .gitignore
├── README.md
├── project_management/
│   ├── 00_项目总规划.md                  # 阶段 1 与阶段 2 总体路线
│   ├── 01_阶段1_引擎搭建_子规划.md        # 1t01–1t04 范围与验收标准
│   ├── tasks/                            # 各 Task 客观执行结果回执
│   └── reviews/                          # 独立审阅记录
├── project_content/
│   ├── config/
│   │   └── sim_config.json               # 唯一标准 JSON 配置文件
│   └── docs/
│       └── 01_撮合与切换确定性规则.md     # 每日 4 步撮合顺序与参数口径
├── src/                                  # 模拟盘核心源码（1t02–1t04 实现）
├── output/                               # 周报与账户状态快照（Git 追踪，不被忽略）
└── tests/                                # 单元测试与只读连通性自检
    └── test_upstream_readonly.py
```

## 4. 运行环境与命令（Conda `quant`）

本仓库强制显式使用 Conda `quant` 环境解释器，禁用系统默认 `python`，严禁执行 `pip install`：

```powershell
# 查看 Python 与 pytest 版本
C:\Users\jxing\.conda\envs\quant\python.exe --version
C:\Users\jxing\.conda\envs\quant\python.exe -m pytest --version

# 运行上游只读连通性自检（pytest）
C:\Users\jxing\.conda\envs\quant\python.exe -m pytest tests/test_upstream_readonly.py -v

# 运行上游只读连通性诊断摘要
C:\Users\jxing\.conda\envs\quant\python.exe tests/test_upstream_readonly.py
```

## 5. 文档导航

- 项目总规划：[project_management/00_项目总规划.md](project_management/00_项目总规划.md)
- 阶段 1 子规划：[project_management/01_阶段1_引擎搭建_子规划.md](project_management/01_阶段1_引擎搭建_子规划.md)
- 唯一配置文件：[project_content/config/sim_config.json](project_content/config/sim_config.json)
- 撮合与切换规则：[project_content/docs/01_撮合与切换确定性规则.md](project_content/docs/01_撮合与切换确定性规则.md)
