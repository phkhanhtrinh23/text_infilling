from collections import OrderedDict
import json
from multiprocessing import cpu_count
import os
import torch
import argparse
from model import Seq2seqVAE
from preprocess import CustomDataset
from torch.utils.data import DataLoader
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def to_word(predictions, idx2word, word2idx):
    sent_str = [str()] * len(predictions)
    for i, sent in enumerate(predictions):
        for id in sent:
            if id == word2idx["<pad>"]:
                break
            sent_str[i] += idx2word[str(id.item())] + " "
    return sent_str

def main(args):
    types = ['train', 'test']

    datasets = OrderedDict()

    for typee in types:
        datasets[typee] = CustomDataset(
            filename=f"{typee}.txt",
            data_dir=args.data_dir,
            data_file=f"{typee}.json",
            file_type=typee,
            max_sequence_length=args.max_sequence_length,
            new_data=False,
        )
    
    with open(os.path.join(args.data_dir, 'vocab.json'), 'r') as file:
        vocab = json.load(file)
    file.close()

    word2idx, idx2word = vocab['word2idx'], vocab['idx2word']

    params = dict(
        vocab_size=len(word2idx),
        sos_idx=word2idx["<sos>"],
        eos_idx=word2idx["<eos>"],
        pad_idx=word2idx["<pad>"],
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
    )

    data_loader = DataLoader(
        dataset=datasets['test'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cpu_count(),
        pin_memory=torch.cuda.is_available()
    )

    if not os.path.exists(args.save_model_path):
        raise FileNotFoundError(args.save_model_path)
    
    model = Seq2seqVAE(**params)
    
    print("Loading model from: ", os.path.join(args.save_model_path))
    model.load_state_dict(torch.load(os.path.join(args.save_model_path)))
    print("Finished.")

    print(model)

    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"
        model = model.to(device)
    
    model.eval()

    list_of_z = torch.Tensor if device == "cpu" else torch.cuda.FloatTensor
    for iteration, batch in enumerate(data_loader):
        batch_size = batch['encoder_input'].size(0)

        for k, v in batch.items():
            batch[k] = v.to(device)

        # ========== Custom z ==========
        input_embedding = model.embedding_1(batch['encoder_input'])
        _, (hidden_state, _) = model.encoder(input_embedding)

        hidden_state = hidden_state.view(batch_size, model.hidden_size * model.hidden_dim) 
        
        mean = model.hidden_to_mean(hidden_state)
        log_variance = model.hidden_to_log_variance(hidden_state)
        std = torch.sqrt(torch.exp(log_variance))
 
        z = mean + std * torch.randn([batch_size, model.latent_size]).to(device)

        if iteration == 0:
            list_of_z = z
        else:
            list_of_z = torch.cat([list_of_z, z], dim=0)

    f = open("output.txt","w")
    for iteration, batch in enumerate(data_loader):
        batch_size = batch['encoder_input'].size(0)

        for k, v in batch.items():
            batch[k] = v.to(device)
        
        """
        1. Shuffle the index
        2. Take the n first elements
        -> Random without replacement
        """
        indices = torch.randperm(len(list_of_z))[:batch_size] 
        z = list_of_z[indices]

        prediction = model.inference(batch_size, batch["encoder_input"], z=z, device=device)
        list_of_inputs = to_word(batch["encoder_input"], idx2word, word2idx)
        list_of_targets = to_word(batch["decoder_target"], idx2word, word2idx)
        list_of_predictions = to_word(prediction, idx2word, word2idx)

        for inp, pred, targ in zip(list_of_inputs, list_of_predictions, list_of_targets):
            f.write(f"Inp: {inp}") # Input
            f.write("\n")
            f.write(f"Pre: {pred}") # Prediction
            f.write("\n")
            f.write(f"Tar: {targ}") # Target
            f.write("\n\n")

        # if iteration == 0:
        #     break
    
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_model_path', type=str, default='checkpoints/model')

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--max_sequence_length', type=int, default=20)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--latent_size', type=int, default=64)

    args = parser.parse_args()

    main(args)