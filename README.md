# UnTool

UnTool是一个用于Sophon芯片推理的Python工具包，支持x86_64和aarch64架构，以及SOC和PCIE两种模式。

## 安装

```bash
pip install untool
```

## 使用示例

**接口使用**：

- 接口1：适用于LLM或多模态LLM，`untool.LLMBasePipeline`封装，便于修改

```python
import argparse
from untool import LLMBasePipeline, MiniCPMVPipeline

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type=str, required=True, help='bmodel文件路径')
parser.add_argument('-t', '--tokenizer_path', type=str, required=True, help='tokenizer文件路径')
parser.add_argument('-d', '--devid', type=int, default=0, help='设备ID')
args = parser.parse_args()

pipeline = LLMBasePipeline(args)  # 常规LLM，如Qwen2.5
# pipeline = MiniCPMVPipeline(args)  # 多模态LLM，如MiniCPMV2.6
pipeline.chat()
```

- 接口2：适用于LLM，C++底层封装，推理速度更快，但不便修改

```python
import argparse
from untool import EngineLLM

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type=str, required=True, help='bmodel文件路径')
parser.add_argument('-t', '--tokenizer_path', type=str, required=True, help='tokenizer文件路径')
parser.add_argument('-d', '--devid', type=int, default=0, help='设备ID')
args = parser.parse_args()

engine = EngineLLM(args)
engine.chat()
```

- 接口3：适用于CV模型等

```python
from untool import EngineOV

model = EngineOV("path/to/model.bmodel", device_id=0)
result = model([input_np_array])[0]
```

`EngineOV`接口可以通过`self.reset_net_stage(net_idx, stage_idx)`切换不同的net和stage。

**基于untool开发的项目参考**：[LLM分布式推理](https://www.modelscope.cn/models/wlc952/TwinStream-LLM)、[AIGC-SDK](https://www.modelscope.cn/models/wlc952/AIGC-SDK)。

## 其他

详见源代码仓库`https://www.modelscope.cn/wlc952/UnTool.git`
