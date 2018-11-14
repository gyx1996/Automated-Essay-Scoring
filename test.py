import sys
import getopt

import data_preprocessing
import model
import score


def main(argv):
    try:
        opts, args = getopt.getopt(argv, '-n:-l:', ['name=', 'loss='])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        sys.exit(2)

    model_name = None
    loss_mode = None
    for o, a in opts:
        if o in ('-n', '--name'):
            model_name = a
        if o in ('-l', '--loss'):
            loss_mode = a
        else:
            assert False, 'unhandled option'

    essays_embedding, labels = (data_preprocessing.generate_essay_embeddings(
        'data/training_set_rel3.tsv',
        'data/word_embedding_glove_6B_200d.txt'))

    if not model_name:
        model_name = 'default'

    if not loss_mode:
        loss_mode = 'CE'

    basic_model = model.Model(name=model_name, loss_mode=loss_mode)
    basic_model.train(essays_embedding, labels)
    y_result = basic_model.test(essays_embedding, labels)
    labels_int = [int(x) for x in labels]
    results_int = [int(x) for x in y_result]
    for i in range(len(labels)):
        print(labels[i], results_int[i], y_result[i])
    print(score.quadratic_weighted_kappa(labels_int, results_int))


if __name__ == '__main__':
    main(sys.argv[1:])
