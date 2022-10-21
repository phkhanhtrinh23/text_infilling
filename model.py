import torch
import torch.nn as nn

class Seq2seqVAE(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, latent_size, max_sequence_length,
                sos_idx, eos_idx, pad_idx, num_layers=1, bidirectional=True):
        super().__init__()

        if torch.cuda.is_available():
            self.tensor = torch.cuda.FloatTensor
        else:
            self.tensor = torch.Tensor

        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.max_sequence_length = max_sequence_length
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding_1 = nn.Embedding(
                            num_embeddings = self.vocab_size,
                            embedding_dim = self.embedding_size
                            )
        
        self.embedding_2 = nn.Embedding(
                            num_embeddings = self.vocab_size,
                            embedding_dim = self.embedding_size
                            )

        self.encoder = nn.LSTM(
                        input_size = self.embedding_size,
                        hidden_size = self.hidden_size,
                        num_layers = self.num_layers,
                        bidirectional = self.bidirectional,
                        batch_first = True
                        )
        
        self.decoder = nn.LSTM(
                        input_size = self.embedding_size,
                        hidden_size = self.hidden_size,
                        num_layers = self.num_layers,
                        bidirectional = self.bidirectional,
                        batch_first = True
                        )
        
        if self.bidirectional:
            self.hidden_dim = 2 * self.num_layers
        else:
            self.hidden_dim = self.num_layers
        
        self.hidden_to_mean = nn.Linear(self.hidden_size * self.hidden_dim, self.latent_size)
        self.hidden_to_log_variance = nn.Linear(self.hidden_size * self.hidden_dim, self.latent_size)
        self.latent_to_hidden = nn.Linear(self.latent_size, self.hidden_size * self.hidden_dim)
        self.input_to_hidden = nn.Linear(self.embedding_size, self.hidden_size * self.hidden_dim)
        self.squeeze_input = nn.Linear(self.max_sequence_length, 1)

        if self.bidirectional:
            self.output_to_vocab = nn.Linear(self.hidden_size * 2, self.vocab_size)
        else:
            self.output_to_vocab = nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward(self, input_sequence, decoder_input_sequence):
        batch_size = input_sequence.size(0)
        input_embedding = self.embedding_1(input_sequence) 
        # input_sequence: I love <mask> football
        # decoder_input_sequence: <sos> I love <mask> football

        _, (hidden_state, _) = self.encoder(input_embedding)

        # flattening matrix
        hidden_state = hidden_state.view(batch_size, self.hidden_size * self.hidden_dim) 
        
        mean = self.hidden_to_mean(hidden_state)
        log_variance = self.hidden_to_log_variance(hidden_state)
        std = torch.exp(0.5 * log_variance)

        z = mean + std * torch.randn([batch_size, self.latent_size]).to(self.device)

        hidden_state = self.latent_to_hidden(z)

        # ========== create conditional input
        input_hidden = self.input_to_hidden(input_embedding)
        # print(input_hidden.shape, hidden_state.shape) # torch.Size([64, 100, 512]) torch.Size([64, 512])
        input_hidden = input_hidden.view(batch_size, self.hidden_size * self.hidden_dim, self.max_sequence_length)
        input_hidden = self.squeeze_input(input_hidden)

        # add conditional input to hidden state
        hidden_state = hidden_state + input_hidden.view(batch_size, self.hidden_size * self.hidden_dim)
        # ==========

        # un-flattening matrix
        hidden_state = hidden_state.view(self.hidden_dim, batch_size, self.hidden_size)

        input_embedding = self.embedding_2(decoder_input_sequence)

        output, (_, _) = self.decoder(input_embedding, (hidden_state, hidden_state))
        
        output = self.output_to_vocab(output)
        # print(output.shape) # torch.Size([64, 100, 30522])

        logits = nn.functional.log_softmax(output, dim=-1)

        return logits, mean, log_variance
    
    def _save_prediction(self, prediction, indices, sequence_running, t):
        temp = prediction[sequence_running]
        # update token at position t
        temp[:,t] = indices.data
        prediction[sequence_running] = temp
        return prediction

    def inference(self, batch_size, input_sequence, z=None, device="cpu"):
        input_embedding = self.embedding_1(input_sequence)
        
        if z is None:
            z = torch.randn([batch_size, self.latent_size]).to(device)
        # else:
        #     batch_size = input_sequence.size(0)
        
        hidden_state = self.latent_to_hidden(z)

        # ==========create conditional input
        input_hidden = self.input_to_hidden(input_embedding)

        input_hidden = input_hidden.view(batch_size, self.hidden_size * self.hidden_dim, self.max_sequence_length)
        input_hidden = self.squeeze_input(input_hidden)

        # add conditional input & hidden state
        hidden_state = hidden_state + input_hidden.view(batch_size, self.hidden_size * self.hidden_dim)
        # ==========

        # un-flattening matrix
        hidden_state = hidden_state.view(self.hidden_dim, batch_size, self.hidden_size)

        # list of indices in a batch_size of sentences => choose which sentence to updates at timestep t
        sequence_idx = torch.arange(0, batch_size).long().to(device)
        sequence_running = torch.arange(0, batch_size).long().to(device)
        sequence_mask = torch.ones(batch_size).bool().to(device)
        running_sequence = torch.arange(0, batch_size).long().to(device)

        prediction = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        cell_state = hidden_state

        t = 0

        while t < self.max_sequence_length and len(running_sequence) > 0:
            if t == 0:
                decoder_input_sequence = self.tensor(batch_size).fill_(self.sos_idx).long().to(device)
            """
            Ex:
            >>> batch_size = 10
            >>> self.sos_idx = 2
            >>> decoder_input_sequence = self.tensor(batch_size).fill_(self.sos_idx).long()
            >>> decoder_input_sequence
            tensor([2,2,2,..,2])
            >>> decoder_input_sequence.shape
            torch.Size([10])
            """
            # print("Decoder ", decoder_input_sequence)
            decoder_input_sequence = decoder_input_sequence.unsqueeze(1)

            """
            Ex:
            >>> decoder_input_sequence = decoder_input_sequence.unsqueeze(1)
            >>> decoder_input_sequence
            tensor([[2],
                    [2],
                    [2],
                    [2],
                    [2],
                    [2],
                    [2],
                    [2],
                    [2],
                    [2]])
            >>> decoder_input_sequence.shape
            torch.Size([10, 1])
            """
            decoder_input_embedding = self.embedding_2(decoder_input_sequence)
            
            # inference
            output, (hidden_state, cell_state) = self.decoder(decoder_input_embedding, (hidden_state, cell_state))
            logits = self.output_to_vocab(output)

            # return list of indices with shape = (batch_size, 1) with each row is top k index
            _, indices = torch.topk(logits, k=1, dim=-1)
            indices = indices.reshape(-1)
            
            prediction = self._save_prediction(prediction, indices, sequence_running, t)

            sequence_mask[sequence_running] = (indices != self.eos_idx)
            sequence_running = sequence_idx.masked_select(sequence_mask)

            running_mask = (indices != self.eos_idx).data
            running_sequence = running_sequence.masked_select(running_mask)

            if len(running_sequence) > 0:
                decoder_input_sequence = indices[running_sequence]
                hidden_state = hidden_state[:, running_sequence]
                cell_state = cell_state[:, running_sequence]

                running_sequence = torch.arange(0, len(running_sequence)).long().to(device)

            t += 1
        
        return prediction