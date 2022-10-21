import os
import csv
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', binary=False)

def create_data():
    for f in ["col_train"]:
        path = f"data/{f}.csv"
        if os.path.exists(path):
            os.remove(path)

    print("Creating data...")

    # ============================== CREATE TRAIN DATA ==============================
    header = ['input', 'label']

    with open("data/col_train.csv", "w", newline='', encoding="utf-8") as f_train:
        writer = csv.writer(f_train)
        writer.writerow(header)
        i = 0

        with open("raw_data/Collocations_Train.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                ele = line.split("\t")
                s, mask_value = ele[-1][1:], ele[-3][:-1]
                s = s.replace("` `", "''")
                if mask_value in model.vocab:
                    list_of_words = model.most_similar(positive=[mask_value], topn=4)
                    for word, cosine_score in list_of_words:
                        inp = s.replace("[MASK]", word)[:-1]
                        lab = s.replace("[MASK]", mask_value)[:-1]
                        writer.writerow([inp, lab])
                        print(f"Loading sentences: {i+1}", end="\r")
                        i += 1
                sent = s.replace("[MASK]", mask_value)[:-1]
                writer.writerow([sent, sent])
                i += 1
        f.close()
        print("\nWriting to file...")

        with open("raw_data/Collocations_Val.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                ele = line.split("\t")
                s, mask_value = ele[-1][1:], ele[-3][:-1]
                s = s.replace("` `", "''")
                if mask_value in model.vocab:
                    list_of_words = model.most_similar(positive=[mask_value], topn=4)
                    for word, cosine_score in list_of_words:
                        inp = s.replace("[MASK]", word)[:-1]
                        lab = s.replace("[MASK]", mask_value)[:-1]
                        writer.writerow([inp, lab])
                        print(f"Loading sentences: {i+1}", end="\r")
                        i += 1
                sent = s.replace("[MASK]", mask_value)[:-1]
                writer.writerow([sent, sent])
                i += 1
        f.close()
        print("\nWriting to file...")

        with open("raw_data/Collocations_Test.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                ele = line.split("\t")
                s, mask_value = ele[-1][1:], ele[-3][:-1]
                s = s.replace("` `", "''")
                if mask_value in model.vocab:
                    list_of_words = model.most_similar(positive=[mask_value], topn=4)
                    for word, cosine_score in list_of_words:
                        inp = s.replace("[MASK]", word)[:-1]
                        lab = s.replace("[MASK]", mask_value)[:-1]
                        writer.writerow([inp, lab])
                        print(f"Loading sentences: {i+1}", end="\r")
                        i += 1
                sent = s.replace("[MASK]", mask_value)[:-1]
                writer.writerow([sent, sent])
                i += 1
        f.close()
        print("\nWriting to file...")

    f_train.close()

    print("Finished.")

if __name__ == "__main__":
    create_data()