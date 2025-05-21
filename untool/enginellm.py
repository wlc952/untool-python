import time
from transformers import AutoTokenizer
from .bindings.wrapper import llm_init, llm_free, llm_forward_first, llm_forward_next, llm_get_seq_len


class EngineLLM():
    def __init__(self, args):
        self.device = args.devid
        self.prompt = {"role":"system", "content":"You are a helpful assistant."}
        self.history = [self.prompt]
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        self.tokenizer.decode([0]) # warm up
        self.ID_EOS = [self.tokenizer.eos_token_id]
        self.llmbase = llm_init(args.model_path, self.device)
        self.SEQLEN = llm_get_seq_len(self.llmbase)
        self.llm_free = llm_free


    def encode_with_tokenizer(self):
        msgs = {'role': 'user', 'content': '{}'.format(self.input_str)}
        self.history.append(msgs)
        self.input_ids = self.tokenizer.apply_chat_template(self.history, tokenize=True, add_generation_prompt=True)
        self.token_len = len(self.input_ids)

    def clear(self):
        self.history = [self.prompt]

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
            self.encode_with_tokenizer()
            print("\nAnswer:")
            
            # Chat
            first_start = time.time()
            token = llm_forward_first(self.llmbase, self.input_ids, self.token_len)
            first_end = time.time()
            tok_num = 1

            full_word_tokens = []
            while token not in self.ID_EOS and self.token_len < self.SEQLEN:
                full_word_tokens.append(token)
                word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
                if "�" not in word:
                    if len(full_word_tokens) == 1:
                        pre_word = word
                        word = self.tokenizer.decode([token, token], skip_special_tokens=True)[len(pre_word):]
                    print(word, flush=True, end="")
                    full_word_tokens = []
                tok_num += 1
                self.token_len += 1
                token = llm_forward_next(self.llmbase)
            next_end = time.time()
            first_duration = first_end - first_start
            next_duration = next_end - first_end
            tps = tok_num / next_duration
            print(f"\nFTL: {first_duration:.3f} s")
            print(f"TPS: {tps:.3f} token/s")
            self.clear()
        
    def close(self):
        if hasattr(self, 'llmbase') and self.llmbase:
            self.llm_free(self.llmbase)
            self.llmbase = None

    def __del__(self):
        self.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the bmodel file')
    parser.add_argument('-t', '--tokenizer_path', type=str, required=True, help='path to the tokenizer file')
    parser.add_argument('-d', '--devid', type=int, default=0, help='device ID to use')
    args = parser.parse_args()

    engine = EngineLLM(args)
    engine.chat()
