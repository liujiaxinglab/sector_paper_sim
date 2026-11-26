# 02｜跨仓路径解析与 CLI 运行说明

> 配置文件：[sim_config.json](../config/sim_config.json) ｜ 执行入口：[run_daily_sim.py](../src/run_daily_sim.py) ｜ 规则口径：[01_撮合与切换确定性规则.md](01_撮合与切换确定性规则.md)

---

## 一、四仓同级工作区约定与路径解析规则

工作区迁移到任意盘符或父目录时，只需保持以下目录处于同一父目录下（无需修改代码或已提交配置）：

```text
<工作区父目录>/
├── quant_research_infrastructure/   # QRI（数据源规范对照）
├── QuantData/                       # 共享数据底座（TDX-01、BS-01、公司行为台账）
├── sector_radar/                    # 上游路由器（watchlist.json、regime_history.json）
└── sector_paper_sim/                # 本仓
```

1. **`upstream_paths` 解析基准**：[sim_config.json](../config/sim_config.json) 中的 5 项上游路径均使用同级仓库相对路径（`../QuantData/...` 与 `../sector_radar/...`）。`load_sim_config()` 会自动根据配置文件位置推导所属仓库根目录（位于 `<repo>/project_content/config/*.json` 时取 `<repo>`，否则取配置文件所在目录），将相对路径统一解析为规范绝对路径后再供 `load_market_bundle()` 与 `run_daily_sim.py` 消费，**不依赖当前工作目录（`cwd`）**。
2. **默认配置文件与默认输出目录**：未显式传 `--config` / `--output-dir` 时，分别默认指向脚本所在仓库的 `project_content/config/sim_config.json` 与 `project_content/output/`。
3. **严禁硬编码个人机器路径**：仓库已提交配置、源码与测试中均不包含盘符绝对路径或个人 Conda 解释器路径；测试子进程统一使用 `sys.executable` 继承当前激活环境解释器。

---

## 二、已核实 CLI 参数与标准运行命令

以下全部命令均要求先激活准备好的 Python >= 3.12 环境（`quant`，含 `numpy`、`pandas`、`pytest`），直接使用当前环境中的 `python`（参见 [本地量化Python执行环境约定.md](../../project_management/common/本地量化Python执行环境约定.md)）。

### 2.1 实测支持参数清单（对应 `run_daily_sim.py` `build_arg_parser`）

| 参数 | 说明 |
| --- | --- |
| `--config <path>` | 模拟配置路径（默认 `project_content/config/sim_config.json`） |
| `--output-dir <path>` | 日频账本输出目录（默认 `project_content/output`） |
| `--as-of YYYY-MM-DD` | 截止交易日（默认取全池最新行情日） |
| `--from-scratch` | 从配置起算日 `start_date` 全量重放 |
| `--allow-lag` | 允许个别标的行情滞后时按 `BS-01` 日历推进 |
| `--dry-run` | 仅计算净值并输出终端摘要，不写磁盘 |
| `--mode {formal,replay}` | 运行模式：`formal`（默认，仅接受时序合格 `LIVE` 决策）或 `replay`（显式历史重放/状态初始化，允许 `BACKFILL`） |

### 2.2 常用运行命令（推荐从 `sector_paper_sim` 仓库根目录执行）

```powershell
# 1. 查看帮助与参数
python project_content/src/run_daily_sim.py --help

# 2. 历史重放演练（读取 BACKFILL 初始化区间，不写磁盘）
python project_content/src/run_daily_sim.py --mode replay --from-scratch --as-of 2026-11-25 --dry-run

# 3. 独立临时目录从头重放（保留正式 output/ 不被改动）
python project_content/src/run_daily_sim.py --mode replay --from-scratch --as-of 2026-11-26 --output-dir <临时输出目录>

# 4. 正式前瞻期日频增量推进（在 sector_radar 完成当日同步与看板后调用）
python project_content/src/run_daily_sim.py

# 5. 从仓库根目录运行全量离线与只读回归测试
python -m pytest -q
```

### 2.3 跨目录调用、测试与外部调度器边界

- **非仓库 `cwd` 调用**：从任意外部目录通过脚本路径调用（例如 `python <工作区父目录>/sector_paper_sim/project_content/src/run_daily_sim.py --mode replay --as-of 2026-11-25 --dry-run`）时，内部默认配置、同级 `QuantData` / `sector_radar` 上游路径及默认输出目录仍精确锚定到所属仓库，不受 `cwd` 影响。
- **跨目录测试**：[pyproject.toml](../../pyproject.toml) 已配置相对 `rootdir` 的 `pythonpath` 与 `testpaths`；从仓库外部目录测试时，可执行：
  ```powershell
  python -m pytest -q -c <工作区父目录>/sector_paper_sim/pyproject.toml --rootdir <工作区父目录>/sector_paper_sim <工作区父目录>/sector_paper_sim/project_content/tests
  ```
- **跨仓日频协同顺序**：先在 `sector_radar` 执行 `--sync-watchlist-regime`（若有当日生效路由切换）与 `run_daily_board.py`，确认 `regime_history.json` 与 `QuantData` 就绪后，再在 `sector_paper_sim` 执行 `run_daily_sim.py`。本仓不创建或修改 Windows 计划任务；外部调度器只需激活环境、设置工作目录并调用上述入口。
