# SR-HGNN
ICDM-2020
《Global Context Enhanced Social Recommendation with Hierarchical Graph Neural Networks》
## Environments

- python 3.8
- pytorch-1.6
- DGL 0.5.3 (https://github.com/dmlc/dgl)

## Example to run the codes		

dataprocess:

```
python dataProcess.py --dataset [CiaoDVD,Epinions,Douban] --rate [0.8,0.6]
```

train SR-HGNN model:

```
python main.py --dataset CiaoDVD --rate 0.8 --batch 128 --lr 0.001 --layer [16,16] --r 0.001 --r2 0.2 --r3 0.01 --dgi_hide_dim 500 --dgi_lr 0.001 --lam_t 0.05 --lam_r 0.1
```


