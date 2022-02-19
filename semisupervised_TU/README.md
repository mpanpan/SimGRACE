## Dependencies

You can create a conda environment named simgrace with the command:
```
conda env create -f environment.yml
conda activate simgrace
```

Then, you need to create two directories for pre-trained models and finetuned results to avoid errors:

```
cd ./pre-training
mkdir models
mkdir logs
cd ..
cd ./funetuning
mkdir logs
cd ..
```

## SimGRACE with Perturbations of Various Magnitudes

Take NCI1 as an example:

### Pre-training: ###

```
cd ./pre-training
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataset NCI1 --eta 1.0 --lr 0.001 --suffix 0
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataset NCI1 --eta 1.0 --lr 0.001 --suffix 1
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataset NCI1 --eta 1.0 --lr 0.001 --suffix 2
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataset NCI1 --eta 1.0 --lr 0.001 --suffix 3
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataset NCI1 --eta 1.0 --lr 0.001 --suffix 4
```

### Finetuning: ###

```
cd ./funetuning
CUDA_VISIBLE_DEVICES=$GPU_ID python main_cl.py --dataset NCI1 --eta 1.0 --semi_split 100 --model_epoch 100 --suffix 0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_cl.py --dataset NCI1 --eta 1.0 --semi_split 100 --model_epoch 100 --suffix 1
CUDA_VISIBLE_DEVICES=$GPU_ID python main_cl.py --dataset NCI1 --eta 1.0 --semi_split 100 --model_epoch 100 --suffix 2
CUDA_VISIBLE_DEVICES=$GPU_ID python main_cl.py --dataset NCI1 --eta 1.0 --semi_split 100 --model_epoch 100 --suffix 3
CUDA_VISIBLE_DEVICES=$GPU_ID python main_cl.py --dataset NCI1 --eta 1.0 --semi_split 100 --model_epoch 100 --suffix 4
```

Five suffixes stand for five runs (with mean & std reported), and eta could be tuned among ```0.1, 1.0, 10.0, 100.0```.
```lr``` in pre-training should be tuned from {0.01, 0.001, 0.0001} and ```model_epoch``` in finetuning (this means the epoch checkpoint loaded from pre-trained model) from {20, 40, 60, 80, 100}.

## Acknowledgements
* https://github.com/Shen-Lab/GraphCL/tree/master/semisupervised_TU
* https://github.com/chentingpc/gfn.
