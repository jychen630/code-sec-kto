# creatiing conda env for halos using the following scripts
# source: https://github.com/ContextualAI/HALOs

```
conda create -n code-sec python=3.10.12
conda activate code-sec
pip3 install numpy==1.24.3 ninja==1.11.1.1 packaging==23.1 
conda install pytorch==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip3 install flash-attn==2.3.3 
pip3 install transformers==4.35.2 datasets hydra-core==1.3.2 wandb==0.15.3 openai==1.6.1 accelerate==0.21.0 tensor-parallel==1.2.4
```

Use `code-sec` conda env with conda-forge channel
```
conda env list
>>>> base                  *  /local/arise/junyao/miniforge3
>>>> code-sec                 /local/arise/junyao/miniforge3/envs/code-sec
```
