import sys
import data_preprocessing
import model


def main(argv):
    essays_embedding, labels = (data_preprocessing.generate_essay_embeddings(
        'data/training_set_rel3.tsv',
        'data/word_embedding_glove_6B_200d.txt'))
    basic_model = model.Model()
    basic_model.train(essays_embedding, labels)
    basic_model.test(essays_embedding, labels)


if __name__ == '__main__':
    main(sys.argv[1:])
