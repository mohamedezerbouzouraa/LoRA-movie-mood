
MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "stanfordnlp/imdb"  
ADAPTER_DIR = "./imdb-lora-adapter"
RESULTS_DIR = "./results"

MAX_LENGTH = 256    
N_TRAIN = 2000      
N_EVAL = 500      
EPOCHS = 1    
SEED = 42

# LoRA settings
LORA_R = 8                           
LORA_ALPHA = 16                   
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_lin", "v_lin"] 

# Training settings
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 5e-4   
WEIGHT_DECAY = 0.01
