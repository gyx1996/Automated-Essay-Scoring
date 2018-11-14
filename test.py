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
    labels_int = [int(x) for x in labels]
    results_int = [int(x) for x in y_result]
    for i in range(len(labels)):
        print(labels[i], results_int[i], y_result[i])
    print(score.quadratic_weighted_kappa(labels_int, results_int))
    print(score.quadratic_weighted_kappa(results_int, labels_int))


if __name__ == '__main__':
    main(sys.argv[1:])
