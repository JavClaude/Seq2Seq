import os
import tqdm
import torch
import mlflow
import logging
import argparse
import configparser

from Utils.data_utils import *
from Tokenizer.tokenizer import train_tokenizer
from Model.nn import Encoder, Decoder, AutoEncoder
from Training.train_eval import train_model, eval_model


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"

# config = configparser.ConfigParser()
# config.read("config.cfg")

# os.environ["MLFLOW_TRACKING_URI"] = config['MLFLOW']["mlflow_tracking_uri"]
# os.environ["MLFLOW_S3_ENDPOINT_URL"] = config['MLFLOW']['mlflow_registry_uri']

# mlflow.set_experiment(config["EXPERIMENT"]["experiment_name"])
# artifact_path = config["EXPERIMENT"]["experiment_name"]

mlflow.set_experiment("AutoEncoder")
artifact_path = "materials"

if not os.path.isdir(".tmp"):
    os.mkdir(".tmp")


def main(**kwargs):

    with open(kwargs["path_to_train_txt"], "r") as file: 
        texts = file.readlines()

        texts_in = list(map(lambda x: x.split("\t")[0], texts))

        with open(".tmp/texts_in.txt", "w") as file_in:
            for text in texts_in:
                file_in.write(text+"\n")

        texts_out = list(map(lambda x: x.split("\t")[1], texts))

        with open(".tmp/texts_out.txt", "w") as file_out:
            for text in texts_out:
                file_out.write(text+"\n")
    
    tokenizer_in = train_tokenizer(**{
        "path_to_text_file": ".tmp/texts_in.txt",
        "num_merges": kwargs.get("vocab_size")
    })

    tokenizer_in.save(".tmp/tokenizer_in.json")
    mlflow.log_artifact(".tmp/tokenizer_in.json", artifact_path=artifact_path)
       
    tokenizer_out = train_tokenizer(**{
        "path_to_text_file": ".tmp/texts_out.txt",
        "num_merges": kwargs.get("vocab_size")
    })

    tokenizer_out.save(".tmp/tokenizer_out.json")
    mlflow.log_artifact(".tmp/tokenizer_out.json", artifact_path=artifact_path)

    kwargs["vocab_in"] = tokenizer_in.get_vocab_size()
    kwargs["vocab_out"] = tokenizer_out.get_vocab_size()
    kwargs.pop("vocab_size")

    train_dataset = NMTDataset(kwargs["path_to_train_txt"], tokenizer_in, tokenizer_out, kwargs["max_sequence_length"], sep="\t")
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=kwargs["batch_size"], sampler=train_sampler)

    test_dataset = NMTDataset(kwargs["path_to_test_txt"], tokenizer_in, tokenizer_out, kwargs["max_sequence_length"], sep="\t")
    test_sampler = torch.utils.data.RandomSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=kwargs["batch_size"], sampler=test_sampler)

    Enc = Encoder(
        vocab_size = tokenizer_in.get_vocab_size(),
        pad_token=tokenizer_in.get_vocab()["<PAD>"],
        **kwargs
    )

    Dec = Decoder(
        vocab_size = tokenizer_out.get_vocab_size(),
        pad_token=tokenizer_out.get_vocab()["<PAD>"],
        **kwargs
    )

    AutoEncoder_Model = AutoEncoder(Enc, Dec, add_noise=True)
    AutoEncoder_Model.to(device)

    Criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_out.get_vocab()["<PAD>"])
    Optimizer = torch.optim.SGD(AutoEncoder_Model.parameters(), lr=kwargs["lr"], momentum=kwargs["momentum"], nesterov=True)

    global_train_it = 0
    global_eval_it = 0

    for epoch in range(kwargs["epochs"]):
        epoch_train_loss, global_train_it = train_model(
            AutoEncoder_Model,
            Optimizer,
            Criterion, 
            train_loader,
            global_train_it
        )

        epoch_valid_loss, global_eval_it = eval_model(
            AutoEncoder_Model,
            Criterion, 
            train_loader,
            global_eval_it
        )

        mlflow.log_metrics({
            "epoch_train_loss": epoch_train_loss,
            "epoch_valid_loss": epoch_valid_loss
        }, step=epoch)
    
    mlflow.pytorch.log_model(AutoEncoder_Model, artifact_path=artifact_path, code_paths=["src/Model"])
    


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--path_to_train_txt", type=str, required=True)
    argument_parser.add_argument("--path_to_test_txt", type=str, required=True)
    argument_parser.add_argument("--vocab_size", type=int, required=False, default=30000)
    argument_parser.add_argument("--embedding_dim", type=int, required=False, default=300)
    argument_parser.add_argument("--hidden_units", type=int, required=False, default=300)
    argument_parser.add_argument("--n_layers", type=int, required=False, default=3)
    argument_parser.add_argument("--bidirectional", type=bool, required=False, default=True)
    argument_parser.add_argument("--dropout", type=float, required=False, default=0.3)
    argument_parser.add_argument("--dropout_rnn", type=float, required=False, default=0.4)
    argument_parser.add_argument("--res_co", type=bool, required=False, default=True)
    argument_parser.add_argument("--lr", type=float, required=False, default=0.002)
    argument_parser.add_argument("--momentum", type=float, required=False, default=0.7)
    argument_parser.add_argument("--max_sequence_length", type=int, required=False, default=64)
    argument_parser.add_argument("--batch_size", type=int, required=False, default=64)
    argument_parser.add_argument("--epochs", type=int, required=False, default=5)
    argument_parser.add_argument("--seed", type=int, required=False, default=42)
    args = argument_parser.parse_args()

    main(**vars(args))