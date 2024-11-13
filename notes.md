accelerate launch \
   launch.py \                                        # main file for launching job
   loss=kto-simple \                                   # must be a file name in config/loss
   model=llama \                                      # must be a file name in config/model
   datasets=[ultrabin,shp] \                          # list of datasets, each with a method (e.g., get_shp) in train/dataloader.py
   exp_name=first_test \                 # experiment name, also the subfolder in cache dir for saving the model          
   ++cache_dir=/local/arise/junyao/huggingface \                               # set the cache directory 
   ++model.name_or_path=meta-llama/Meta-Llama-3-8B \        # HF (or local) repo containing model configs, vocab, etc.
   ++model.load_from=/local/arise/junyao/huggingface/hub/models--codellama--CodeLlama-7b-hf \    # load existing model as starting point; if empty, use model.name_or_path
   ++lr=5e-6 \                                              # set the learning rate
   ++loss.beta=0.1 


python train.py loss=kto-simple model=codellama7b datasets=[ultrabin,shp] exp_name=first_test ++cache_dir=/local/nlp/junyao/huggingface ++model.name_or_path=codellama/CodeLlama-7b-hf ++model.load_from=/local/nlp/junyao/huggingface/hub/models--codellama--CodeLlama-7b-hf ++lr=5e-6 ++loss.beta=0.1


pkill -9 -u $USER python && python train.py loss=kto model=codellama7b datasets=[ultrabin] exp_name=first_test ++cache_dir=/local/nlp/junyao/huggingface ++model.name_or_path=codellama/CodeLlama-7b-hf  ++lr=5e-6 ++loss.beta=0.1 ++model.batch_size=1 


pkill -9 -u $USER python
HF_token=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


When find ineffective changes, clean cache


This line is slow: 
conda install pytorch==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia


cuda error might be related to the data indexing
https://discuss.pytorch.org/t/runtimeerror-cuda-error-device-side-assert-triggered-index-out-of-bounds-failed/87827/6

using bitsandbytes for quantization temporarily suppress the cuda out of memory error



==================================================================================
4 & 5 crafted prompts words
def get_ultrabin(split: str, human_prefix: str, human_suffix: str, assistant_prefix: str, assistant_suffix: str) -> Dataset:
    """
    Load the Ultrafeedback (binarized) dataset from Huggingface and convert it into to a Dataset.
    For this dataset, the SFT text is the preferred response.

    Args:
        - split: one of 'test', 'train'
        - human_prefix: marks start of human turn ('<|user|>' is the recommended choice and is set in config.yaml)
        - human_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)
        - assistant_prefix: marks start of human turn ('<|assistant|>' is the recommended choice and is set in config.yaml)
        - assistant_suffix: marks end of human turn ('' is the recommended choice and is set in config.yaml)

    Returns:   
        A Dataset instance.
    """
    
    data = Dataset('ultrabin')

    #Example 1: Simple prompt with chosen/rejected responses
    prompt1 = "<|user|>What is Python?<|assistant|>"
    data[prompt1].prompt = prompt1
    data[prompt1].generations = [
        "Python is a high-level programming language known for its readability.<|assistant|>",
        "Python is just a snake that programmers talk about.<|assistant|>"
    ]
    data[prompt1].pairs = [(0, 1)]  # First response preferred over second
    data[prompt1].sft_index = 0     # First response is the best one
    data[prompt1].dataset_name = 'ultrabin'
    data[prompt1].truncation_mode = 'keep_start'
    data[prompt1].remove_extra_spaces()

    prompt1 = "3<|user|>What is Python?<|assistant|>"
    data[prompt1].prompt = prompt1
    data[prompt1].generations = [
        "Python is a high-level programming language known for its readability.<|assistant|>",
        "Python is just a snake that programmers talk about.<|assistant|>"
    ]
    data[prompt1].pairs = [(0, 1)]  # First response preferred over second
    data[prompt1].sft_index = 0     # First response is the best one
    data[prompt1].dataset_name = 'ultrabin'
    data[prompt1].truncation_mode = 'keep_start'
    data[prompt1].remove_extra_spaces()

    
    prompt1 = "2<|user|>What is Python?<|assistant|>"
    data[prompt1].prompt = prompt1
    data[prompt1].generations = [
        "Python is a high-level programming language known for its readability.<|assistant|>",
        "Python is just a snake that programmers talk about.<|assistant|>"
    ]
    data[prompt1].pairs = [(0, 1)]  # First response preferred over second
    data[prompt1].sft_index = 0     # First response is the best one
    data[prompt1].dataset_name = 'ultrabin'
    data[prompt1].truncation_mode = 'keep_start'
    data[prompt1].remove_extra_spaces()

    
    prompt1 = "4<|user|>What is Python?<|assistant|>"
    data[prompt1].prompt = prompt1
    data[prompt1].generations = [
        "Python is a high-level programming language known for its readability.<|assistant|>",
        "Python is just a snake that programmers talk about.<|assistant|>"
    ]
    data[prompt1].pairs = [(0, 1)]  # First response preferred over second
    data[prompt1].sft_index = 0     # First response is the best one
    data[prompt1].dataset_name = 'ultrabin'
    data[prompt1].truncation_mode = 'keep_start'
    data[prompt1].remove_extra_spaces()

config.yaml
eval_every: 20_000
n_samples: 1 #128
n_eval_examples: 2 #512
reference_dtype: bfloat16
max_grad_norm: 10.0
v_head_max_grad_norm: 0.10
max_length: 2048
max_prompt_length: 1024
activation_checkpointing: true
batch_size: 32
gradient_accumulation_steps: 1
eval_batch_size: 2 # 16 # hanging for 16, work for 4
use_flash_attention: false

=======================================================
eval_batch_size: 2,4,8 works; 16 not work



### how to write readme
https://github.com/eth-sri/sven


export CUDA_VISIBLE_DEVICES to env var in train.sh works; but not in train.py using os.environ["CUDA_VISIBLE_DEVICES"]


pickle err: https://discuss.huggingface.co/t/cant-pickle-error-using-accelerate-multi-gpu/32358/4


using quantization_config=None (i.e 16bits) works so that forward pickle error is not triggered
but cuda out of memory


batch_size: 4, eval_batch_size: 4 solve cuda out of memory error for codellama7b

20min, 400 examples, 2 gpus
16000 examples -> 14 hours


pip default package installation
# method1 
pip install --target=/local/nlp/junyao/code-sec/packages pyfiglet

# method2
export PYTHONPATH=/local/nlp/junyao/code-sec/packages:$PYTHONPATH
pip install --prefix=/local/nlp/junyao/code-sec pyfiglet

# copy paste the following in the terminal. dont just ./greene.sh
export PATH="/scratch/jc9723/miniconda3/bin:$PATH"
conda init
source /scratch/jc9723/miniconda3/etc/profile.d/conda.sh
source activate code-sec
conda activate code-sec # not recommended, and not leveraging packages installled


## install packgaes
# CUDA setup
export CUDA_HOME=/usr/local/cuda  # adjust this path to your CUDA installation
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

tmux new -t test


# launch srun interactive session
srun --gres=gpu:1 --mem=48GB --time=48:00:00 --account=pr_177_general --pty bash


mv default hugginface token storage to your large-disk storage

mv /home/jc9723/.cache/huggingface/token /scratch/jc9723/huggingface/token

using srun to configure env.
then request sbatch 


remembert o laod all data

remember to increate eval batch size from 2 to 32 (originally 128 & 512)


distributed training sigsegv:
https://discuss.pytorch.org/t/how-to-fix-a-sigsegv-in-pytorch-when-using-distributed-training-e-g-ddp/113518


srun --gres=gpu:a100:1 --mem=80GB --nodes=1 --time=48:00:00 --account=pr_177_general --pty bash



## check block name
### starcoder
import inspect
from transformers.models.gpt_bigcode import GPTBigCodeModel
print(inspect.getsource(GPTBigCodeModel))
>>> GPTBigCodeBlock

### phi2
import inspect
from transformers.models.phi import PhiModel
print(inspect.getsource(PhiModel))
>>> PhiDecoderLayer


### codellama7b
>>> LlamaDecoderLayer


## transformers version
codellama7b, starcoderbase: transformers==4.35.2
phi2: transformers==4.37.0