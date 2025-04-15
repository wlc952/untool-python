# UnTool

UnTool是一个用于Sophon芯片推理的Python工具包，支持x86_64和aarch64架构，以及SOC和PCIE两种模式。

## 安装

```bash
pip install untool
```

## 使用示例

```python
from untool import EngineLLM
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the bmodel file')
parser.add_argument('-t', '--tokenizer_path', type=str, required=True, help='path to the tokenizer file')
parser.add_argument('-d', '--devid', type=int, default=0, help='device ID to use')
parser.add_argument('--generation_mode', type=str, choices=["greedy", "penalty_sample"], default="greedy", help='mode for generating next token')
parser.add_argument('--enable_history', action='store_true', help="if set, enables storing of history memory")
args = parser.parse_args()

engine = EngineLLM(args)
engine.chat()
```

```python
from untool import EngineOV
net = EngineOV("rmbg.bmodel", device_id=0)

# Prepare input
image = preprocess_image(orig_image, model_input_size)

# Inference 
result = net([image])[0]

# Post process    
result_image = postprocess_image(result, orig_im_size)

```

## 其他
源代码仓库`https://www.modelscope.cn/wlc952/UnTool.git`