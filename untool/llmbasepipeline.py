import os
import time
import torch
from .llmbasemodel import LLMBaseModel, MiniCPMV
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image


class LLMBasePipeline:
    def __init__(self, args):
        self.model_path = args.model_path
        self.device = getattr(args, "devid", 0)
        self.enable_history = getattr(args, "enable_history", False)
        self.generation_mode = getattr(args, "generation_mode", "greedy")
        self.quant_type = getattr(args, "quant_type", "bf16")
        self.system_prompt = "You are a helpful assistant."
        self.history = [{"role": "system", "content": self.system_prompt}]

        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        self.tokenizer.decode([0]) # warm up
        self.EOS = self.tokenizer.eos_token_id
        self.model = LLMBaseModel(model_path=self.model_path, device_id=self.device, quant_type=self.quant_type, generation_mode=self.generation_mode)

    def clear(self):
        self.history = [{"role": "system", "content": self.system_prompt}]


    def update_history(self):
        if self.model.token_length >= self.model.SEQLEN:
            print("... (reach the maximal length)", flush=True, end="")
            self.history = [{"role": "system", "content": self.system_prompt}]
        else:
            self.history.append({"role": "assistant", "content": self.answer_cur})


    def encode_with_tokenizer(self):
        self.history.append({"role": "user", "content": self.input_str})
        input_ids = self.tokenizer.apply_chat_template(self.history, tokenize=True, add_generation_prompt=True)
        return input_ids


    def chat(self):
        print(
            """\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
================================================================="""
        )
        while True:
            self.input_str = input("\nQuestion: ")
            if self.input_str in ["exit", "q", "quit"]:
                break
            elif self.input_str in ["clear", "new"]:
                self.clear()
            else:
                tokens = self.encode_with_tokenizer()
                if not tokens:
                    print("Sorry: your question is empty!!")
                    return
                if len(tokens) > self.model.SEQLEN:
                    print("The maximum question length should be shorter than {} but we get {} instead.".format(self.model.SEQLEN, len(tokens)))
                    return
                print("\nAnswer: ", end="")
                self.stream_answer(tokens)


    def stream_answer(self, tokens):
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []

        # First token
        first_start = time.time()
        token = self.model.forward_first(tokens)
        first_end = time.time()
        # Following tokens
        full_word_tokens = []
        while token != self.EOS and self.model.token_length < self.model.SEQLEN:
            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
            if "�" not in word:
                if len(full_word_tokens) == 1:
                    pre_word = word
                    word = self.tokenizer.decode([token, token], skip_special_tokens=True)[len(pre_word):]
                print(word, flush=True, end="")
                self.answer_token += full_word_tokens
                full_word_tokens = []
            tok_num += 1
            token = self.model.forward_next()

        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration

        if self.enable_history:
            self.answer_cur = self.tokenizer.decode(self.answer_token)
            self.update_history()
        else:
            self.clear()

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")


class MiniCPMVPipeline:
    def __init__(self, args):
        self.model_path = args.model_path
        self.device = getattr(args, "devid", 0)
        self.generation_mode = getattr(args, "generation_mode", "greedy")
        self.quant_type = getattr(args, "quant_type", "bf16")
        self.processor = AutoProcessor.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.decode([0])
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.ID_EOS = [self.tokenizer.eos_token_id, self.ID_IM_END]
        self.MAX_SLICE_NUMS = self.processor.image_processor.max_slice_nums
        self.model = MiniCPMV(model_path=self.model_path, device_id=self.device, quant_type=self.quant_type, generation_mode=self.generation_mode)
    
    def encode_with_image(self):
        inserted_image_str = "(<image>./</image>)\n"
        images = []
        contents = []
        for i in range(self.patch_num): 
            images.append(Image.open(self.image_str[i]).convert('RGB').resize((448, 448), Image.LANCZOS))
            contents.append(inserted_image_str)
        contents.append(self.input_str)

        msgs = [{'role': 'user', 'content': ''.join(contents)}]
        prompts_lists = self.processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            prompts_lists,
            [images],
            max_slice_nums=self.MAX_SLICE_NUMS,
            use_image_id=None,
            return_tensors="pt",
            max_length=8192
        )
        self.input_ids = inputs.input_ids[0]
        self.pixel_values = torch.cat(inputs["pixel_values"][0], dim=0).flatten().tolist()
        self.image_bound = inputs["image_bound"][0]
        self.image_offsets = [idx for start, end in self.image_bound.tolist() for idx in range(start, end)]
        self.input_ids = self.input_ids.tolist()


    def encode(self):
        msgs = [{'role': 'user', 'content': '{}'.format(self.input_str)}]
        prompts_lists = self.processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            prompts_lists,
            [[]],
            return_tensors="pt",
            max_length=8192
        )
        self.image_offsets = []
        self.pixel_values = []
        self.patch_num = 0
        self.input_ids = inputs.input_ids[0].tolist()


    def chat(self):
        print(
            """\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
================================================================="""
        )
        while True:
            self.input_str = input("\nQuestion: ")
            if self.input_str in ["exit", "q", "quit"]:
                break
            try:
                self.patch_num = int(input("\nImage Num (0 ~ 14): "))
            except:
                self.patch_num = 0
            self.image_str = [input(f"\nImage Path {i}: ") for i in range(self.patch_num)] if self.patch_num >= 1 else []

            if self.image_str:
                missing_images = [x for x in self.image_str if not os.path.exists(x)]
                if missing_images:
                    print("Missing images: {}".format(", ".join(missing_images)))
                    continue
                else:
                    self.encode_with_image()
            else:
                self.encode()

            print("\nAnswer:")
            first_start = time.time()
            token = self.model.forward_first(self.input_ids, self.pixel_values, self.image_offsets, self.patch_num)
            first_end = time.time()
            tok_num = 1
            full_word_tokens = []
            while token not in self.ID_EOS and self.model.token_length < self.model.SEQLEN:
                full_word_tokens.append(token)
                word = self.tokenizer.decode(
                    full_word_tokens, skip_special_tokens=True)
                if "�" not in word:
                    if len(full_word_tokens) == 1:
                        pre_word = word
                        word = self.tokenizer.decode([token, token], skip_special_tokens=True)[len(pre_word):]
                    print(word, flush=True, end="")
                    full_word_tokens = []
                tok_num += 1
                token = self.model.forward_next()
            next_end = time.time()
            first_duration = first_end - first_start
            next_duration = next_end - first_end
            tps = tok_num / next_duration
            print(f"\nFTL: {first_duration:.3f} s")
            print(f"TPS: {tps:.3f} token/s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the bmodel file')
    parser.add_argument('-t', '--tokenizer_path', type=str, default="../support/token_config", help='path to the tokenizer file')
    parser.add_argument('-d', '--devid', type=int, default=0, help='device ID to use')
    parser.add_argument('--generation_mode', type=str, choices=["greedy", "penalty_sample"], default="greedy", help='mode for generating next token')
    parser.add_argument('--enable_history', action='store_true', help="if set, enables storing of history memory")
    args = parser.parse_args()

    # pipline = LLMBasePipeline(args)
    pipline = MiniCPMVPipeline(args)
    pipline.chat()