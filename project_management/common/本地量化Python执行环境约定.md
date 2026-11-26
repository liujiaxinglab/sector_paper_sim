# 本地量化 Python 执行环境约定

本项目所有本地 Python 脚本、自动化单测及自检命令，**统一使用当前已激活的 `quant` Python 环境（Python >= 3.12，已预装 `numpy`、`pandas`、`pytest`）中的 `python`**：

```text
python --version
```

## 强制要求

1. **禁止硬编码个人本机绝对解释器路径**：文档、脚本与测试中严禁写死某台电脑的用户目录或 Conda 安装绝对路径；需先激活本机准备好的 `quant`（或对应依赖齐全的）Python 环境再执行命令，子进程内部统一使用 `sys.executable` 继承当前运行解释器（Windows 下未激活环境时切勿误调系统 `WindowsApps` 占位 `python.exe`）。
2. **禁止擅自换环境或装包**：若遇到解释器不可用、依赖包缺失或版本异常，**立即中止执行并原样上报报错信息**，严禁自行切换其他 Python 环境，严禁执行 `pip install` 或 `conda install`。
3. **文件编码与换行统一**：本仓全部文本、文档、配置与源码文件统一使用 **UTF-8 无 BOM** 编码与 **LF（`\n`）** 换行符。

## PowerShell 标准调用示例

在 `sector_paper_sim` 仓库根目录执行（已激活 `quant` 环境）：

```powershell
python --version
python project_content/src/run_daily_sim.py --help
python -m pytest -v
```