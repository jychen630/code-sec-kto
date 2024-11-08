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


python train.py loss=kto model=codellama7b datasets=[ultrabin] exp_name=first_test ++cache_dir=/local/nlp/junyao/huggingface ++model.name_or_path=codellama/CodeLlama-7b-hf  ++lr=5e-6 ++loss.beta=0.1 ++model.batch_size=1 


pkill -9 -u $USER python