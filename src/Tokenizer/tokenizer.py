import argparse
import tokenizers

def main(**kwargs):
    pass

def train_tokenizer(**kwargs):
    tokenizer = tokenizers.ByteLevelBPETokenizer()
    tokenizer.train(files=kwargs["path_to_text_file"], vocab_size=kwargs["num_merges"])
    tokenizer.add_special_tokens(["<EOS>", "<SOS>", "<PAD>"])
    return tokenizer

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--path_to_text_file", type=str, required=True, default=False)
    argument_parser.add_argument("--num_merges", type=int, required=False, default=30000)

    args = argument_parser.parse_args()

    main(**vars(args))