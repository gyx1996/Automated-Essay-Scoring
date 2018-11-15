import sys
import getopt

import data_preprocessing
import model
import score


def main(argv):
    try:
        opts, args = getopt.getopt(
            argv,
            '-n:-l:-t',
            ['name=', 'loss=', 'train'])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    model_name = None
    loss_mode = None
    train_mode = False
    for o, a in opts:
        if o in ('-n', '--name'):
            model_name = a
        elif o in ('-l', '--loss'):
            loss_mode = a
        elif o in ('-t', '--train'):
            train_mode = True
        else:
            assert False, 'unhandled option'

    valid_essays, valid_labels = data_preprocessing.get_valid_essay_embeddings(
        'data/valid_set.tsv',
        'data/valid_sample_submission_2_column.csv',
        'data/word_embedding_glove_6B_200d.txt')

    if not model_name:
        model_name = 'Default'

    if not loss_mode:
        loss_mode = 'CE'

    print('Begin...')
    basic_model = model.Model(name=model_name, loss_mode=loss_mode)
    if train_mode:
        essays_embedding, labels = \
            data_preprocessing.get_train_essay_embeddings(
                'data/training_set_rel3.tsv',
                'data/word_embedding_glove_6B_200d.txt')
        basic_model.train(essays_embedding, labels)
    y_result = basic_model.test(valid_essays, valid_labels)
    labels_int = [int(x) for x in valid_labels]
    results_int = [int(x) for x in y_result]
    for i in range(len(valid_labels)):
        print(valid_labels[i], results_int[i], y_result[i])
    print(score.quadratic_weighted_kappa(labels_int, results_int))


if __name__ == '__main__':
    main(sys.argv[1:])
