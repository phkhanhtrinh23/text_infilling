from collections import Counter, OrderedDict, defaultdict
import copy
import os
import json
import numpy as np
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer

class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

class CustomDataset(Dataset):
    def __init__(self, filename, max_sequence_length, data_file, file_type = "train", data_dir="data/", new_data=True):
        super().__init__()
        self.filename = filename
        self.max_sequence_length = max_sequence_length
        self.file_type = file_type
        self.data_dir = data_dir
        self.data_file = data_file
        self.tokenizer = TweetTokenizer(preserve_case=False)
        self.vocab_file = "vocab.json"
        self.new_data = new_data
        
        if self.new_data is True and os.path.exists(os.path.join(self.data_dir, self.data_file)):
            os.remove(os.path.join(self.data_dir, self.data_file))

        if os.path.exists(os.path.join(self.data_dir, self.data_file)) is False:
            print(f"Creating new vocab and input for {self.file_type} data...")
            self._create_data()
            self._load_data()
        else:
            print("Loading existing vocab and input...")
            self._load_data()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'encoder_input': np.asarray(self.data[idx]['encoder_input']),
            'decoder_input': np.asarray(self.data[idx]['decoder_input']),
            'decoder_target': np.asarray(self.data[idx]['decoder_target']),
            # 'length': self.data[idx]['length']
        }
    
    @property
    def vocab_size(self):
        return len(self.word2idx)
    
    @property
    def pad_idx(self):
        return self.word2idx['<pad>']

    @property
    def sos_idx(self):
        return self.word2idx['<sos>']

    @property
    def eos_idx(self):
        return self.word2idx['<eos>']
    
    @property
    def mask_idx(self):
        return self.word2idx['<mask>']

    @property
    def unk_idx(self):
        return self.word2idx['<unk>']

    def _create_vocab(self):
        counter = OrderedCounter()
        word2idx = dict()
        idx2word = dict()

        special_tokens = ['<pad>', '<mask>', '<unk>', '<sos>', '<eos>']
        for token in special_tokens:
            idx2word[len(word2idx)] = token
            word2idx[token] = len(word2idx)

        with open(os.path.join(self.data_dir, self.filename), "r") as file:
            for _, line in enumerate(file):
                words = self.tokenizer.tokenize(line)
                counter.update(words)

            for word, _ in counter.items():
                if word not in special_tokens:
                    idx2word[len(word2idx)] = word
                    word2idx[word] = len(word2idx)

        assert len(word2idx) == len(idx2word)

        vocab = dict(word2idx=word2idx, idx2word=idx2word)
        with open(os.path.join(self.data_dir, self.vocab_file), 'wb') as file:
            vocab = json.dumps(vocab, ensure_ascii=False)
            file.write(vocab.encode('utf-8', 'replace'))

        self._load_vocab()
    
    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
            vocab = json.load(file)

        self.word2idx, self.idx2word = vocab['word2idx'], vocab['idx2word']

    def _create_data(self):
        if os.path.exists(os.path.join(self.data_dir, self.vocab_file)):
            print("Loading existing vocab file...")
            self._load_vocab()
            print("Finised.")
        else:
            self._create_vocab()

        data = defaultdict(dict)
        with open(os.path.join(self.data_dir, self.filename), "r") as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                input_line, target_line = line.split("\t\t")
                input_tokens = self.tokenizer.tokenize(input_line)
                target_tokens = self.tokenizer.tokenize(target_line)
                
                encoder_input = input_tokens
                encoder_input = encoder_input[:self.max_sequence_length]

                decoder_input = ['<sos>'] + target_tokens
                decoder_input = decoder_input[:self.max_sequence_length]

                decoder_target = target_tokens[:self.max_sequence_length - 1]
                decoder_target = decoder_target + ['<eos>']

                length = len(input_tokens)

                encoder_input.extend(['<pad>'] * (self.max_sequence_length - length))
                decoder_input.extend(['<pad>'] * (self.max_sequence_length - length - 1))
                decoder_target.extend(['<pad>'] * (self.max_sequence_length - length - 1))

                encoder_input = [self.word2idx.get(word, self.word2idx["<unk>"]) for word in encoder_input]
                decoder_input = [self.word2idx.get(word, self.word2idx["<unk>"]) for word in decoder_input]
                decoder_target = [self.word2idx.get(word, self.word2idx["<unk>"]) for word in decoder_target]

                idx = len(data)
                data[idx]["encoder_input"] = encoder_input
                data[idx]["decoder_input"] = decoder_input
                data[idx]["decoder_target"] = decoder_target
                # data[idx]["length"] = length

                print(f"Loading sentences: {i+1}", end="\r")
            
        if os.path.exists(os.path.join(self.data_dir, self.data_file)):
            os.remove(os.path.join(self.data_dir, self.data_file))

        print("\nWriting to file...")

        with open(os.path.join(self.data_dir, self.data_file), 'wb') as file:
            # for chunk in json.JSONEncoder().iterencode(data):
            #     file.write(chunk)
            data = json.dumps(data, ensure_ascii=False)
            file.write(data.encode('utf-8', 'replace'))
        file.close()

        print("Finished")
    
    def _load_data(self):
        with open(os.path.join(self.data_dir, self.data_file), "r") as file:
            self.data = json.load(file)
        file.close()

        self._load_vocab()