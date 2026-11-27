# 2t03｜撮合引擎五项规则修正与日频 CLI 实现

覆盖规划：[2 子规划.md](2%20子规划.md) 的 **2.3（`2.3.1`、`2.3.2`）**。
当前状态：**执行已完成，见 [2t03-执行结果.md](2t03-执行结果.md)；阶段 2 主控终审待通过，阶段 3 尚未启动**。

---

## 目标

全面修正初版撮合与报告层的 5 项核心规则及可复现性缺陷（清零 `ISS-003`、`ISS-004`、`ISS-005`、`ISS-006`、`ISS-007`、`ISS-009`）：严格落地 [01_撮合与切换确定性规则.md](../../../project_content/docs/01_撮合与切换确定性规则.md) 定稿的单日 4 步时序（`TRANSITION` 开盘清仓、分红复投标准佣金、网格 Step 2 `P0 = open_T` + 50% 底仓 + `0.015` 下限等差网格 + K 线阴阳路径、海龟无 `1% ATR` 纯 `1/6` 预算 `T+1` 开盘执行、单标的停牌顺延），废除 `run_weekly_sim.py` 与墙钟绑定状态，上线 `run_daily_sim.py`、精简 `state.json`、`nav_daily.csv`、`nav_latest.md` 与 `history/nav_YYYYMMDD.md`，并完成全部单元测试重写。

## 精确输入

- [01_撮合与切换确定性规则.md](../../../project_content/docs/01_撮合与切换确定性规则.md)（第一至第四节全部契约）
- [路线决策记录.md](../../路线决策记录.md)（`RD-001`–`RD-004`）
- [issues.md](../../issues.md)（`ISS-003`–`ISS-007`、`ISS-009`）
- `2t02` 交付后的 [sim_config.json](../../../project_content/config/sim_config.json) 与 `project_content/src/data_loader.py`
- 待重构源码与测试：`project_content/src/{strategies,engine,reporter}.py`、待废除替换的 `project_content/src/run_weekly_sim.py`、`project_content/tests/test_{engine_rules,reporter_and_cli}.py`
- [本地量化Python执行环境约定.md](../../common/本地量化Python执行环境约定.md)

## 执行步骤

1. **重构 `project_content/src/strategies.py`**：
   - **清理免佣与 `1% ATR` 违规分支**：移除 `max_affordable_shares` 中的 `commission_free` 逻辑（统一强制万二/最低 `0.10` 元佣金）；彻底删除 `TurtleConfig.risk_fraction` 及 `compute_turtle_risk_shares`，改写海龟判定为基于 `T-1` 收盘相对此前 55 日最高收盘价突破 / 20 日最低收盘价破位、并于 `T` 日 Step 2 按 `open_T` 成交，买入纯按 `min(prev_equity / 6, available_cash)` 整手定仓；
   - **重构日频等差网格**：实现以 `P0 = open_T` 建 50% 底仓、锁定步长 `s = max(ATR20_{T-1} / close_{T-1}, min_step_pct)`（`min_step_pct = 0.015`，前复权 `T-1` 口径）与等差档位 `P0 * (1 ± k * s)`；重写 `match_grid_bar` 支持基于 K 线阴阳的确定性路径（阳线/平盘 `open -> low -> high -> close`、阴线 `open -> high -> low -> close`），确保当日新买份额受 T+1 锁定不可当日卖出、当日新生成的反向挂单于次交易日生效。
2. **重构 `project_content/src/engine.py`**：
   - 将每交易日 4 步时序严格对齐定稿规则：Step 1 盘前拆分与除息现金到账及含佣 `open_T` 复投；Step 2 `open_T` 路由切换（切入 `TRAP_OFF` 与 `TRANSITION` 均 100% 清仓）、跨态平仓、`RANGE_GRID` 开盘建网、海龟 `T+1` 开盘执行及 `C`/`D` 组建仓；Step 3 盘中等差网格撮合；Step 4 收盘估值、双守恒校验；
   - 实现单标的停牌日顺延机制：停牌当日除息照常入账，开盘复投/清仓/建网/海龟单据保存待执行意图至复牌首个交易日开盘执行，盘中跳过，收盘沿用最近前收估值并记录停牌标记；
   - 精简 `export_state()` / `from_state()`：仅导出截面断点状态（账户现金、权益、净值、峰值/最大回撤、持仓、策略挂单与顺延队列、`next_trade_id`、`last_known_closes`），不再将全量历史 `trades`、`daily_records`、`excess_history`、`corporate_actions` 塞入断点结构；
   - `Account A` 四态每日由 `regime_history.json`（`effective_date <= T`）确定性解析，彻底删除对墙钟和外部锁定缓存的依赖。
3. **重构 `project_content/src/reporter.py` 与新建 `project_content/src/run_daily_sim.py`（删除 `run_weekly_sim.py`）**：
   - `reporter.py` 实现原子写入：精简 `state.json`、`trades.csv`（支持增量合并去重按 `(date, account, int(trade_id))` 排序）、`nav_daily.csv`（每日 5 账户净值/回撤/超额结构化长表或宽表按 `date` 幂等追加更新）、`nav_latest.md` 与 `history/nav_YYYYMMDD.md`（按日生成不可覆盖的当日净值快照）；
   - `run_daily_sim.py` 实现 `--config`、`--output-dir`、`--as-of`、`--from-scratch`、`--allow-lag`、`--dry-run`（起算日固定源自配置 `2026-11-23`，彻底废除 `next_effective_regime` 与 `run_weekly_sim.py`）；
   - `--from-scratch` 模式下从零重建并原子更新输出产物；默认增量模式下从 `state.json` 断点与既有 `nav_daily.csv` / `trades.csv` 增量推进至 `--as-of`。
4. **重写规则与 CLI 单元测试**：
   - 重写 `project_content/tests/test_engine_rules.py` 与 `project_content/tests/test_reporter_and_cli.py`，彻底废除旧错断言（如原免佣测试、`TRANSITION` 保留僵尸仓测试、1% ATR 缩仓测试、收盘建网测试），新增覆盖全部 5 项新规、K 线阴阳路径、停牌日顺延、精简 `state.json` 行数控制、`--from-scratch` vs 逐日增量等价性的完备单测。
5. **测试验证与客观回执**：全量运行 `pytest` 确认全部通过，撰写 `2t03-执行结果.md` 并更新 [status.md](../../status.md)。

## 交付位置

- `project_content/src/strategies.py`、`project_content/src/engine.py`、`project_content/src/reporter.py`、`project_content/src/run_daily_sim.py`（并删除旧 `run_weekly_sim.py`）
- `project_content/tests/test_engine_rules.py`、`project_content/tests/test_reporter_and_cli.py`
- `project_management/subplan/2 日频化与合规重构/2t03-执行结果.md`、[status.md](../../status.md)

## 验收依据

1. `ISS-003`–`ISS-007`、`ISS-009` 对应的全部 5 项撮合修正、无墙钟状态、精简 `state.json` 与 `nav_daily.csv`/`nav_latest.md`/`history/nav_YYYYMMDD.md` 均在代码中严格落地且旧 `run_weekly_sim.py` 已清理；
2. 合成数据下全量重算 `--from-scratch` 与逐日增量运行产出的 `nav_daily.csv`、`trades.csv` 与期末持仓 100% 一致；
3. 全量 `pytest` 在任意工作目录下执行全部通过（退出码 `0`），零外部仓库写入。
