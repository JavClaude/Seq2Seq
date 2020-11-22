import torch
import logging
from tqdm import tqdm
from tokenizers import Tokenizer
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"


class NMTDataset(Dataset):
    def __init__(self,
                path_to_text_file: str,
                tokenizer_in: Tokenizer,
                tokenizer_out: Tokenizer,
                max_sequence_length: int,
                sep: str,
                **kwargs):
        
        logger.info("Processing file: {}".format(path_to_text_file))

        self.pad_token_in = tokenizer_in.get_vocab()['<PAD>']
        self.pad_token_out = tokenizer_out.get_vocab()['<PAD>']
        self.max_sequence_length = max_sequence_length

        with open(path_to_text_file, "r") as file:
            texts = file.readlines()

        texts = list(map(lambda x: x.split(sep), texts)) 
        texts = list(map(lambda x: x[0:2], texts)) 

        self.texts_in = []
        self.texts_in_length = []
        self.texts_out = []

        for i in tqdm(range(len(texts)), desc="Tokenization...."):
            text_in_ids = tokenizer_in.encode(texts[i][0]).ids
            texts_out_ids = tokenizer_out.encode(texts[i][1]).ids

            if len(text_in_ids) and len(texts_out_ids) <= max_sequence_length:
                self.texts_in.append(text_in_ids)
                self.texts_in_length.append(len(text_in_ids))
                self.texts_out.append(texts_out_ids)

        logger.info("# Texts: {}".format(len(self.texts_in)))

    def __len__(self):
        return len(self.texts_in)

    def __getitem__(self, index: int):
        sequence_in = torch.full(size=(self.max_sequence_length,), fill_value=self.pad_token_in, dtype=torch.long, device=device)
        sequence_length = torch.tensor(self.texts_in_length[index], dtype=torch.long, device=device)
        sequence_out = torch.full(size=(self.max_sequence_length,), fill_value=self.pad_token_out, dtype=torch.long, device=device)

        for i, id in enumerate(self.texts_in[index]):
            sequence_in[i] = id

        for i, id in enumerate(self.texts_out[index]):
            sequence_out[i] = id

        return (sequence_in, sequence_length, sequence_out)
