# UnTool

UnTool是一个用于Sophon芯片推理的Python工具包，支持x86_64和aarch64架构，以及SOC和PCIE两种模式。

## 安装

```bash
pip install untool
```

## 使用示例

```python
import untool

# 可选: 明确设置模式
untool.set_mode('pcie')  

# 使用库功能
tensor = untool.untensor_create()
runtime = untool.runtime_init("model.bmodel", 0)
# 其他操作...
```

## 其他
源代码仓库`https://www.modelscope.cn/wlc952/UnTool.git`