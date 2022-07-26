from pathlib import Path
import transformers
from torch.cuda.amp import GradScaler

# class Config:
#     NB_EPOCHS = 2
#     LR = 3e-4
#     T_0 = 20
#     η_min = 1e-4
#     MAX_LEN = 120
#     TRAIN_BS = 16
#     VALID_BS = 16
#     MODEL_NAME = 'bert-large-uncased'
#     data_dir = Path('../input/AI4Code')
#     TOKENIZER = transformers.BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)
#     scaler = GradScaler()


class Config:
    NB_EPOCHS = 15
    LR = 3e-5
    T_0 = 20
    η_min = 1e-4
    MD_MAX_LEN = 120
    TOTAL_MAX_LEN = 512
    TRAIN_BS = 32
    VALID_BS = 32
    MODEL_NAME = 'microsoft/codebert-base'
    data_dir = Path('../input/')
    output_dir = Path(f'./outputs/{MODEL_NAME}')
    # TOKENIZER = transformers.BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)
    scaler = GradScaler()
    wb_key = '254c36ecc9968daa5343f5047af6160cc9d791da'
    n_workers = 8
