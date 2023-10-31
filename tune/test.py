from peft import LoraConfig, TaskType
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# TRANSFORM=`python -c "import transformers;print('/'.join(transformers.__file__.split('/')[:-1])+'/models/llama/convert_llama_weights_to_hf.py')"`
# python ${TRANSFORM} --input_dir llama-2-13b --model_size 13B --output_dir llama-hf/13b

MODEL_PATH = "../llama-hf/7b"
# MODEL_PATH = "../llama-hf/13b"

# tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
# model = LlamaForCausalLM.from_pretrained(MODEL_PATH)

# prompt = "Yo mama so fat"
# inp = tokenizer(prompt, return_tensors="pt").to("cuda")
# print(inp["input_ids"].device)

# model.eval()
# model.to("cuda")
# with torch.no_grad():
#     out = model.generate(**inp)

# out.shape
# print(tokenizer.decode(out[0]))

print(torch.__version__)
