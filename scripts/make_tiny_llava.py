"""Create a tiny RANDOM-WEIGHT LLaVA (same tokenizer/processor/image-token layout as
llava-hf/llava-1.5-7b-hf: 336px, patch 14 -> 576 visual tokens, layer -2, 'default' strategy)
for CPU end-to-end validation of the pruning path. Usage: make_tiny_llava.py OUT_DIR"""
import sys
import torch
from transformers import (AutoProcessor, CLIPVisionConfig, LlamaConfig, LlavaConfig,
                          LlavaForConditionalGeneration)

out = sys.argv[1]
src = sys.argv[2] if len(sys.argv) > 2 else "llava-hf/llava-1.5-7b-hf"
proc = AutoProcessor.from_pretrained(src)
vc = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_attention_heads=2, num_hidden_layers=3,
                      image_size=336, patch_size=14, projection_dim=32)
tc = LlamaConfig(vocab_size=32064, hidden_size=64, intermediate_size=128, num_attention_heads=4,
                 num_key_value_heads=4, num_hidden_layers=2, max_position_embeddings=4096,
                 pad_token_id=32001, bos_token_id=1, eos_token_id=2)
cfg = LlavaConfig(vision_config=vc, text_config=tc, image_token_index=32000, vision_feature_layer=-2,
                  vision_feature_select_strategy="default", projector_hidden_act="gelu", pad_token_id=32001)
torch.manual_seed(0)
m = LlavaForConditionalGeneration(cfg).eval()
m.save_pretrained(out)
proc.save_pretrained(out)
print("saved", out, sum(p.numel() for p in m.parameters()), "params")
