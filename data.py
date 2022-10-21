import os
from random import randrange
# from random import randrange
from nltk.tokenize import TweetTokenizer
import string

def create_train_valid_data():
    for f in ["train", "valid"]:
        path = f"data/{f}.txt"
        if os.path.exists(path):
            os.remove(path)

    print("Creating data...")

    tokenizer = TweetTokenizer(preserve_case=False)

    # ============================== CREATE TRAIN DATA ==============================
    f_train = open("data/train.txt", "w")
    with open("raw_data/pos_train.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            words = tokenizer.tokenize(line)
            if len(words) <= 3:
                i = -1
            else:
                i = randrange(len(words))
            text = [word if idx != i or word in string.punctuation else "<mask>" for idx, word in enumerate(words)]
            s = " ".join(text)
            s = s.replace("` `", "''")
            line = line.replace("``", "''")
            f_train.write(s + "\t\t" + line)
    f.close()

    with open("raw_data/neg_train.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            words = tokenizer.tokenize(line)
            if len(words) <= 3:
                i = -1
            else:
                i = randrange(len(words))
            text = [word if idx != i or word in string.punctuation else "<mask>" for idx, word in enumerate(words)]
            s = " ".join(text)
            s = s.replace("` `", "''")
            line = line.replace("``", "''")
            f_train.write(s + "\t\t" + line)
    f.close()
    f_train.close()

    # ============================== CREATE VALID DATA ==============================
    f_valid = open("data/valid.txt", "w")
    with open("raw_data/pos_valid.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            words = tokenizer.tokenize(line)
            if len(words) <= 3:
                i = -1
            else:
                i = randrange(len(words))
            text = [word if idx != i or word in string.punctuation else "<mask>" for idx, word in enumerate(words)]
            s = " ".join(text)
            s = s.replace("` `", "''")
            line = line.replace("``", "''")
            f_valid.write(s + "\t\t" + line)
    f.close()

    f_valid.write("\n")

    with open("raw_data/neg_valid.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            words = tokenizer.tokenize(line)
            if len(words) <= 3:
                i = -1
            else:
                i = randrange(len(words))
            text = [word if idx != i or word in string.punctuation else "<mask>" for idx, word in enumerate(words)]
            s = " ".join(text)
            s = s.replace("` `", "''")
            line = line.replace("``", "''")
            f_valid.write(s + "\t\t" + line)
    f.close()
    f_valid.close()

    print("Finished.")

def create_test_data():
    path = f"data/test.txt"
    if os.path.exists(path):
        os.remove(path)

    print("Creating data...")

    tokenizer = TweetTokenizer(preserve_case=False)

    # ============================== CREATE TEST DATA ==============================
    # ============================== CREATE ONE TIME ONLY ==============================
    f_test = open("data/test.txt", "w")
    with open("raw_data/pos_test.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            words = tokenizer.tokenize(line)
            if len(words) <= 3:
                i = -1
            else:
                i = randrange(len(words))
            text = [word if idx != i or word in string.punctuation else "<mask>" for idx, word in enumerate(words)]
            s = " ".join(text)
            s = s.replace("` `", "''")
            line = line.replace("``", "''")
            f_test.write(s + "\t\t" + line)
    f.close()

    with open("raw_data/neg_test.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            words = tokenizer.tokenize(line)
            if len(words) <= 3:
                i = -1
            else:
                i = randrange(len(words))
            text = [word if idx != i or word in string.punctuation else "<mask>" for idx, word in enumerate(words)]
            s = " ".join(text)
            s = s.replace("` `", "''")
            line = line.replace("``", "''")
            f_test.write(s + "\t\t" + line)
    f.close()
    f_test.close()

    print("Finished.")

if __name__ == "__main__":
    # create_train_valid_data()
    create_test_data()