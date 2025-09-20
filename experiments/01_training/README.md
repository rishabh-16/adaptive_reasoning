# Training Instructions

## Setup

   1. Create and activate a conda environment:
   ```bash
   conda create -n 01_training python=3.12
   conda activate 01_training
   ```
   
   2. Install the package:


   ```bash
   cd LLaMA-Factory
   pip install -e ".[torch,metrics]" --no-build-isolation
   pip install deepspeed==0.16.9
   pip install liger-kernel
   ```


Logs:
File to edit: /home/rishabhtiwari/.conda/envs/01_training/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py

- 30B-multi: 2 nodes, 100000 examples, dynamic
- 1024563: 2 nodes, 100000 examples, topk=8

- 102664 and 1110609: instruct model 8 nodes, 1.2 million examples, topk=8, output dir in llama factory
- 1110828: instruct model 8 nodes, 1.2 million examples, topk=16

![Dynamic Training](dynamic.png) 
1111568: instruct model 8 nodes, 1.2 million examples, topk=[8,12,16,20]


1115206: base model 8 nodes, 1.2 million examples, top k = 8
1115438: base model 8 nodes, 1.2 million examples, topk=[8,12,16,20]







