# 本地量化 Python 执行环境约定

本项目所有本地 Python 脚本、自动化单测及自检命令，**强制显式使用 Conda `quant` 环境解释器**：

```text
C:/Users/jxing/.conda/envs/quant/python.exe
```

## 强制要求

1. **禁止使用系统 PATH 默认 `python`**：所有命令行调用与子进程启动，必须显式指向 `C:/Users/jxing/.conda/envs/quant/python.exe`（Windows 下对应 `C:\Users\jxing\.conda\envs\quant\python.exe`）。
2. **禁止擅自换环境或装包**：若遇到解释器不可用、依赖包缺失或版本异常，**立即中止执行并原样上报报错信息**，严禁自行切换其他 Python 环境，严禁执行 `pip install` 或 `conda install`。
3. **文件编码与换行统一**：本仓全部文本、文档、配置与源码文件统一使用 **UTF-8 无 BOM** 编码与 **LF（`\n`）** 换行符。

## PowerShell 标准调用示例

```powershell
$Py = 'C:\Users\jxing\.conda\envs\quant\python.exe'
& $Py --version
& $Py -m pytest -v
```
