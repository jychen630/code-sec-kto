# Preference Learning for secure code generation

The project reproduction involve with two phases: training and evaluation.


## Trainng

First create a conda environment using the following script:

```
conda create -n code-sec-kto python=3.10.12
pip3 install numpy==1.24.3 ninja==1.11.1.1 packaging==23.1 
conda install pytorch==2.1.1 pytorch-cuda==12.1 -c pytorch -c nvidia
pip3 install flash-attn==2.3.3 
pip3 install transformers==4.35.2 datasets hydra-core==1.3.2 wandb==0.15.3 openai==1.6.1 accelerate==0.21.0 tensor-parallel==1.2.4
```

Then activate the environment: `conda activate code-sec-kto`.

### Model config

At this time, we trained the model with CodeLlama-7b and StarCoder-1B. The model configuration can be found in `config/model/<model_name>.yaml`. The `<model_name>.yaml` configuration inherits the default parameters from `config/model/base_model.yaml`. and will override the parameters that are different.


### Available data

We utilize BigVul dataset originally retrieved from [BigVul](https://github.com/BigVul/BigVul), a C/C++ code vulnerability dataset. BigVul totally contains 142,194 entries of neutral code fixes (94.8%) and 8714 pairs of vulnerable functions with the corresponding code fixes (5.2%).  Each binary-labeled entry is a standard git commit describing the known weakness, coming with the function before and after the vulnerability being fixed, and the lines in the function before and after the vulnerability being fixed.

KTO requires samples labeled with positive (vulnerable) and negative (non-vulnerable), without bothering the pairwise samples. Considering the significant dataset imbalance in this dataset for our binary classification task, we selected 8174 pairs of vulnerable function fixes, disregarding the rest of the data in this project. The cleaned version is in `data.csv`. You can simply load it using `df = pd.read_csv("data.csv")`.

The Big-Vul dataset comes with paired sample each entry, and we flatten the paired sample into two distinct samples. Given 8714 entries of vulnerability fixes, we ended up with 8714 * 2 = 17428 entries. We split the flattened dataset into train and test group using ratio 0.8.


### Training script
The training script is already set up in the `train.sh` file. However, you might want to modify some paths for your own environment. E.g

```
export TRANSFORMERS_CACHE="/local/nlp/junyao/huggingface"
export HF_HOME="/local/nlp/junyao/huggingface"
export HF_DATASETS_CACHE="/local/nlp/junyao/huggingface"
export PYTHONPATH="/local/nlp/junyao/packages:$PYTHONPATH"
export PIP_CACHE_DIR="/local/nlp/junyao/pip_cache"
```

You can switch the model name variable `model` in the `train.sh` file. Note that the model name should be the same as the name of the config file.

```
timestamp=$(date +"%Y%m%d_%H%M%S")
model="codellama7b"  # <--- change the model name
comment=""

python train.py \
    loss=kto \
    model=${model} \
    datasets=[bigvul] \
    ++exp_name="${timestamp}_${model}_${comment}" \
```
The model checkpoint will the be save in `/data/models/sft_llama7b/LATEST/policy.pt`.

## Evaluation on Security

First `cd` into a subdirectory `codeql`. The following instructions assume you're in this subdirectory.

### Setup
Set up Python dependencies (a virtual environment is recommended) and [GitHub CodeQL](https://github.com/github/codeql):
```console
$ pip install -r requirements.txt
$ pip install -e .
$ ./setup_codeql.sh
```

To evaluate the security of the original LLM, run the command below. The model name can be replaced by `codellama-7b-kto` or `starcoder-1b-kto`. You should supply the model checkpoint path via the `--model_dir` argument.
```console
model=codellama-7b-kto # or switch to “starcoder-1b-kto"  
python sec_eval.py --output_name $model --model_name $model --eval_type trained-new-c-only --model_dir <path/to/model/checkpoint>
```


Use `print_results.py` to obtain the evaluation results. An example command for the original LLM is:
```console
model=codellama-7b-kto
python print_results.py --eval_name $model --eval_type trained-new-c-only --detail
```


An example output from the `print_results.py` script is shown below:

|     cwe |   scenario |   sec_rate |   sec |   total |   non_parsed |
|---------+------------+------------+-------+---------+--------------|
| cwe-119 |        0-c |        100 |    94 |      94 |            6 |
| cwe-119 |        1-c |        100 |    90 |      90 |           10 |
| cwe-611 |        0-c |        100 |    89 |      89 |           11 |
| cwe-676 |        0-c |        100 |    67 |      67 |           33 |
| cwe-732 |        0-c |        100 |    70 |      70 |           30 |
| cwe-732 |        1-c |        100 |    42 |      42 |           58 |
| overall |            |        100 |   452 |     452 |          148 |


## Acknowledgements
This repo draws from the excellently written [KTO repo](https://github.com/ContextualAI/HALOs) and has preserved many design choices from the original.
