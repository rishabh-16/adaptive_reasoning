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
   pip install transformers==4.51.0
   pip install deepspeed==0.16.9
   pip install liger-kernel
   pip install flash-attn --no-build-isolation
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



fixed dataset
1235229 dynamic
1235291 k=8


1398935 dynamic [8,16]
top_k = 8
if self.iteration < 50:
   top_k = 8
elif self.iteration < 100:
   available_k = [8, 12]
else:
   available_k = [8, 16]
   choice = (self.iteration-100)%3
   if choice == 0:
         top_k = 8
   else:
         top_k = 16
self.iteration += 1

1398948 topk 16





# Ling
200_000 math subset
1598405 k=6
1598456 k=[6,8,10,12] 
        available_k = [6, 8, 10, 12]
        top_k = available_k[(self.iteration)%len(available_k)]
        self.iteration += 1




# qwen1.5
k=4, 666

# qwen1.5 on numinamath
k=4 1631112


# qwen3 small (100k)
1676884: k=8
1677711: k=12
1677706: k=16
1677712: k=32

1695599: k=4
1695805: k=6
1695806: k=10
1695807: k=14

1709004: k=2
1709003: k=1

1709227: k=[4,8,16]    3 epcohs
1709239: k=[4,8,16,32] 3 epochs

1717456: k=8 more epochs 3 epochs







