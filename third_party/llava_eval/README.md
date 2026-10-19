Reference files from the official LLaVA-1.5 evaluation (haotian-liu/LLaVA, Apache-2.0).
- *.jsonl.gz question files and llava-v1.5-13b answer files: from LLaVA's `eval.zip`
  (HF mirror Xiaodong/LLaVA-1.5-Eval, sha256 2df36a33e3d3947d3e351e627449ee7cc7725d69ecc8dcbb2103d90172fbe17a).
- *.py.txt: copies of llava/eval/{eval_pope,eval_textvqa,model_vqa_loader}.py, llava/{conversation,mm_utils}.py (main) for reference.
Used by scripts/audit_llava_protocol.py and tests/test_protocol.py.
