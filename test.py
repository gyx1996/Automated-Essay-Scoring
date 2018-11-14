import sys
import data_preprocessing
import model
import score


def main(argv):
    essays_embedding, labels = (data_preprocessing.generate_essay_embeddings(
        'data/training_set_rel3.tsv',
        'data/word_embedding_glove_6B_200d.txt'))
    basic_model = model.Model()
    # basic_model.train(essays_embedding, labels)
    y_result = basic_model.test(essays_embedding, labels)
    print(score.quadratic_weighted_kappa(labels, y_result))
    print(score.quadratic_weighted_kappa(y_result, labels))


if __name__ == '__main__':
    main(sys.argv[1:])
