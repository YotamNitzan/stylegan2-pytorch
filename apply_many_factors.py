import os
from tqdm import tqdm
from pathlib import Path

d = 3
n = 5
# ckpt = '/home/yotam/models/stylegans/official/stylegan2-car-config-f.pt'
# factor_file = '/home/yotam/projects/ganstudent/sefa/my_own/cars_factor.pt'
# out_dir = '/home/yotam/projects/ganstudent/sefa/my_own/results_layer'
# factor_dir = '/home/yotam/projects/stylegan2-pytorch/layer_factors'

ckpt = '/home/yotam/models/stylegans/official/afhqcat-mirror-paper512-ada-resumeffhq512-freezed13.pt'
factor_file = '/home/yotam/projects/ganstudent/sefa/cats/factors/factors.pt'
out_dir = '/home/yotam/projects/ganstudent/sefa/cats/results/'

iters=15

for i in tqdm(range(iters)):
    os.system(f'/mnt/data/yotamnitzan/anaconda3/envs/psp/bin/python apply_single_layer_factor.py'
              f'  -i {i} -d {d} -n {n} --ckpt {ckpt}  --out_dir {out_dir} --size 512 {factor_file}')

# for factor_file in Path(factor_dir).iterdir():
#     try:
#         layer_num = int(factor_file.name.split('-')[1])
#     except Exception as e:
#         continue
#
#     for i in tqdm(range(iters)):
#         # os.system(f'/mnt/data/yotamnitzan/anaconda3/envs/psp/bin/python apply_factor.py'
#         #           f'  -i {i} -d {d} -n {n} --ckpt {ckpt}  --out_dir {out_dir} --size 512 {factor_file}')
#         os.system(f'/mnt/data/yotamnitzan/anaconda3/envs/psp/bin/python apply_factor.py'
#                   f'  -i {i} -d {d} -n {n} --ckpt {ckpt}  --out_dir {out_dir} --size 512 {str(factor_file)}'
#                   f' --layer {layer_num}')
