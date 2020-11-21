import torch
from torch.nn.utils.rnn import pad_packed_sequence

device = "cuda" if torch.cuda.is_avaible() else "cpu"


class Encoder(torch.nn.Module):
    def __init__(self,
                vocab_size: int,
                pad_token: int, 
                embedding_dim: int, 
                hidden_units: int,
                n_layers: int,
                bidirectional: bool,
                dropout_rnn: float,
                **kwargs):
        super(Encoder, self).__init__()

        self.pad_token = pad_token
    
        self.hidden_units = hidden_units
        self.total_hiddens = hidden_units * 2 if bidirectional else hidden_units
        self.n_layers = n_layers
        self.n_directions = n_layers * 2 if bidirectional else n_layers

        self.Embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.LN = torch.nn.LayerNorm(embedding_dim)
        self.Lstm = torch.nn.LSTM(embedding_dim, hidden_units, n_layers, batch_first=True, dropout=dropout_rnn)
    
    def forward(self, sequences, sequences_length):
        sequences = self.Embedding(sequences)
        sequences = self.LN(sequences)
        packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(sequences, sequences_length, batch_first=True, enforce_sorted=False)
        packed_sequences, hiddens = self.Lstm(packed_sequences)
        sequences, _ = pad_packed_sequence(packed_sequences)
        return sequences, hiddens

class Decoder(torch.nn.Module):
    def __init__(self,
                vocab_size: int,
                embedding_dim: int,
                hidden_units: int,
                n_layers: int,
                bidirectional: bool,
                res_co: bool,
                dropout: float,
                dropout_rnn: float,
                **kwargs):
        super(Decoder, self).__init__()

        self.res_co = res_co
        self.hidden_units = hidden_units
        self.total_hiddens = hidden_units * 2 if bidirectional else hidden_units
        self.n_layers = n_layers
        self.n_directions = n_layers * 2 if bidirectional else n_layers

        self.Embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.LN = torch.nn.LayerNorm(embedding_dim)
        self.Lstm = torch.nn.LSTM(embedding_dim, hidden_units, n_layers, batch_first=True, dropout=dropout_rnn, bidirectional=bidirectional)

        if res_co:
            if embedding_dim != self.total_hiddens:
                self.do_lin_proj = True
                self.Res_co = torch.nn.Sequential(
                    torch.nn.Linear(embedding_dim, self.total_hiddens),
                    torch.nn.ReLU()
                )
            else:
                self.do_lin_proj = False
        else:
            pass

        self.Linear = torch.nn.Linear(self.total_hiddens, vocab_size)            
        self.Dropout = torch.nn.Dropout(dropout)
    
    def forward(self, tokens, hiddens):
        tokens = self.Embedding(tokens)
        tokens = self.LN(tokens)
        hidden_tokens, hiddens = self.Lstm(tokens, hiddens)

        if self.res_co:
            if self.do_lin_proj:
                hidden_tokens = torch.add(self.Res_co(tokens), hidden_tokens.squeeze())
            else:
                hidden_tokens = torch.add(tokens, hidden_tokens.squeeze())

        hidden_tokens = self.Linear(hidden_tokens)
        hidden_tokens = self.Dropout(hidden_tokens)

        return hidden_tokens, hiddens


class AutoEncoder(torch.nn.Module):
    def __init__(self,
                Encoder: Encoder,
                Decoder: Decoder,
                **kwargs):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder
        self.decoder = Decoder

        # Do assert
        assert self.encoder.total_hiddens == self.decoder.total_hiddens, "Encoder and Decoder must have the same number of hidden units"
        assert self.encoder.n_directions == self.decoder.n_directions, "Encoder and Decoder must have the same number of direction"

    def forward(self, inputs_sequence, inputs_length, targets_sequence):
        pass






