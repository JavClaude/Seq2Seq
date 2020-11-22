import tqdm
import torch
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(Model, Optimizer, Criterion, Dataset, global_train_it):
    Model.train()
    Model.zero_grad()

    local_train_it = 0
    epoch_loss = 0

    for batch in tqdm.tqdm(Dataset):
        batch = tuple(t.to(device) for t in batch)
        seq_in, seq_in_length, seq_out = batch
        
        preds = Model(seq_in, seq_in_length, seq_out)

        loss = Criterion(preds[:, 1:].transpose(2, 1), seq_out[:, 1:])
        loss.backward()

        Optimizer.step()
        Model.zero_grad()

        epoch_loss += loss.item()
        local_train_it += 1
        global_train_it += 1

    epoch_loss /= local_train_it

    return epoch_loss, global_train_it


def eval_model(Model, Criterion, Dataset, global_eval_it):
    Model.eval()

    local_eval_it = 0
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm.tqdm(Dataset):
            batch = tuple(t.to(device) for t in batch)
            seq_in, seq_in_length, seq_out = batch
            
            preds = Model(seq_in, seq_in_length, seq_out)

            loss = Criterion(preds[:, 1:].transpose(2, 1), seq_out[:, 1:])

            epoch_loss += loss.item()
            local_eval_it += 1
            global_eval_it += 1

    epoch_loss /= local_eval_it

    return epoch_loss, global_eval_it
