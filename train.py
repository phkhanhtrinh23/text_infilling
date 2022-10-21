import os
import json
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict
from model import Seq2seqVAE
from preprocess import CustomDataset
from multiprocessing import cpu_count
import numpy as np
from scheduler import CosineWithWarmRestarts
from data import create_train_valid_data

def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step-x0))))
    elif anneal_function == 'linear':
        return min(1, step / x0)

def loss_function(NLL, logp, target, mean, logv, anneal_function, step, k, x0):

    # Flatten 'target'
    target = target.contiguous().view(-1)

    # Resize prediction from 3D to 2D
    logp = logp.view(-1, logp.size(-1))

    # Negative Log-likelihood Loss
    NLLLoss = NLL(logp, target.long())

    # # KL Divergence
    KLLoss = (-0.5) * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    KLWeight = kl_anneal_function(anneal_function, step, k, x0)

    return NLLLoss, KLLoss, KLWeight

def get_length(train):
    for i, _ in enumerate(train):
        pass
    return i

def main(args):
    all_types = ['train', 'valid', 'test']
    types = ['train', 'valid']

    datasets = OrderedDict()

    for typee in all_types:
        datasets[typee] = CustomDataset(
            filename=f"{typee}.txt",
            data_dir=args.data_dir,
            data_file=f"{typee}.json",
            file_type=typee,
            max_sequence_length=args.max_sequence_length,
        )

    params = dict(
        vocab_size=datasets['train'].vocab_size,
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
    )

    model = Seq2seqVAE(**params)
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"
        model = model.to(device)

    print(model)

    save_model_path = args.save_model_path
    if os.path.exists(save_model_path) is False:
        os.makedirs(save_model_path)

    with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    NLL = torch.nn.NLLLoss(ignore_index=datasets['train'].pad_idx, reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    step = 0

    train_iter = DataLoader(
                dataset=datasets['train'],
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )
    train_len = get_length(train_iter)

    if args.scheduler == True:
        sched = CosineWithWarmRestarts(optimizer, T_max=train_len)

    min_current_loss = float("inf")

    if os.path.exists(os.path.join(args.logdir, "log.txt")):
        os.remove(os.path.join(args.logdir, "log.txt"))

    f_log = open(os.path.join(args.logdir, "log.txt"), "w")
    
    num_early_stopping = 0
    
    for epoch in range(args.epochs):

        if (epoch > 0) and (epoch % args.num_generate_data == 0):
            print(f"Generate new data every {args.num_generate_data} epochs")
            f_log.write(f"Generate new data every {args.num_generate_data} epoch\n")

            create_train_valid_data()

            for typee in types:
                datasets[typee] = CustomDataset(
                    filename=f"{typee}.txt",
                    data_dir=args.data_dir,
                    data_file=f"{typee}.json",
                    file_type=typee,
                    max_sequence_length=args.max_sequence_length,
                )

        for typee in types:
            data_loader = DataLoader(
                dataset=datasets[typee],
                batch_size=args.batch_size,
                shuffle=(typee=='train'),
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            if typee == 'train':
                model.train()
            else:
                model.eval()
            
            current_loss = []

            print(f"{typee.upper()}\nEpoch {epoch+1}/{args.epochs}:")
            f_log.write(f"{typee.upper()}\nEpoch {epoch+1}/{args.epochs}:\n")

            for iteration, batch in enumerate(data_loader):
                batch_size = batch['encoder_input'].size(0)
                for k, v in batch.items():
                    batch[k] = v.to(device)

                # Forward pass
                logp, mean, logv = model(batch['encoder_input'], batch['decoder_input'])

                # Loss calculation
                NLLLoss, KLLoss, KLWeight = loss_function(NLL, 
                                                        logp, 
                                                        batch['decoder_target'],
                                                        mean, 
                                                        logv, 
                                                        args.anneal_function, 
                                                        step, 
                                                        args.k, 
                                                        args.x0)
                loss = (NLLLoss + KLWeight * KLLoss) / batch_size

                # Backpropagation & Optimization
                if typee == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1
                    if args.scheduler == True: 
                        sched.step()

                current_loss.append(loss.item())

                if iteration % args.print_every == 0 or (iteration + 1) == len(data_loader):
                    print("Batch: %i/%i, Loss: %.4f, NLL-Loss: %.4f, KL-Loss: %.4f, KL-Weight: %.4f" % 
                        (iteration, len(data_loader)-1, loss.item(), NLLLoss.item()/batch_size, KLLoss.item()/batch_size, KLWeight))
                    f_log.write("Batch: %i/%i, Loss: %.4f, NLL-Loss: %.4f, KL-Loss: %.4f, KL-Weight: %.4f\n" % 
                        (iteration, len(data_loader)-1, loss.item(), NLLLoss.item()/batch_size, KLLoss.item()/batch_size, KLWeight))
            
            mean_loss = np.mean(current_loss)
            print("Mean Loss: %.4f" % mean_loss)
            f_log.write("Mean Loss: %.4f\n" % mean_loss)

            # Save checkpoint
            if typee == 'valid' and mean_loss < min_current_loss:
                min_current_loss = mean_loss
                num_early_stopping = 0
                checkpoint_path = os.path.join(save_model_path, "model")
                torch.save(model.state_dict(), checkpoint_path)
                print("Best model saved at: %s" % checkpoint_path)
                f_log.write("Best model saved at: %s\n" % checkpoint_path)

                prediction = model.inference(batch_size, batch["encoder_input"], device=device)
                print("Prediction ", prediction)
                print("Target ", batch["decoder_target"])
                # print("Logp ", torch.argmax(logp, dim=-1))
                # print("Target ", batch["decoder_target"])
            elif mean_loss > min_current_loss and args.early_stopping == True:
                num_early_stopping += 1
                if num_early_stopping == args.num_early_stopping:
                    print(f"Early Stopping after {args.num_early_stopping} epochs!")
                    raise Exception
        print()
        f_log.write("\n")

    f_log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--max_sequence_length', type=int, default=20)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--scheduler', type=bool, default=True)

    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--bidirectional', type=bool, default=False) 
    """
        Recommendation: I chose not to use Bidirectional-LSTM because 
        the results are not satisfying.
    """
    parser.add_argument('--latent_size', type=int, default=64)

    parser.add_argument('--anneal_function', type=str, default='logistic')
    parser.add_argument('--k', type=float, default=0.0032)
    parser.add_argument('--x0', type=int, default=3200)

    parser.add_argument('--early_stopping', type=bool, default=False)
    parser.add_argument('--num_early_stopping', type=int, default=10)
    parser.add_argument('--print_every', type=int, default=100) # print every batch
    parser.add_argument('--logdir', type=str, default='logs/')
    parser.add_argument('--save_model_path', type=str, default='checkpoints/')

    parser.add_argument('--num_generate_data', type=int, default=2)

    args = parser.parse_args()

    args.anneal_function = args.anneal_function.lower()

    assert args.anneal_function in ['logistic', 'linear']

    main(args)