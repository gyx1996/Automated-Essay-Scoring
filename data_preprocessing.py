"""Generate data from given file path.

Author:
    Yixu Gao

Usage:
    essays_embedding, labels = generate_essay_embeddings(
        'data/training_set_rel3.tsv',
        'data/word_embedding_glove_6B_200d.txt')
"""
from nltk.tokenize import regexp_tokenize
from numpy.random import uniform
from numpy import reshape


def load_train_data(filename,
                    headline=True,
                    target_essay_set=1,
                    lowercase=True):
    """Load data from .tsv file.

    Args:
        filename: string
        headline: bool
        target_essay_set: int in {1, 2, ..., 8}
        lowercase: bool

    Returns:
        essays and scores from given target essay set
        essays: 1-D string list
        labels: 1-D int list
    """
    essays = []
    labels = []
    with open(filename, errors='ignore') as fd:
        if headline:
            fd.readline()
        lines = fd.readlines()
    for line in lines:
        splits = line.split('\t')
        essay_set = int(splits[1])
        if target_essay_set == essay_set:
            if lowercase:
                essay = splits[2].strip('"').lower()
            else:
                essay = splits[2].strip('"')
        else:
            essay = ''
        if essay_set == 1:
            essays.append(essay)
            labels.append(int(splits[6]) - 2)
    return essays, labels


def load_valid_data(essay_file,
                    label_file,
                    headline=True,
                    target_essay_set=1,
                    lowercase=True):
    """Load data from .tsv file.

    Args:
        essay_file: string, data/valid_set.tsv
        label_file: string, data/valid_sample_submission_2_column.tsv
        headline: bool
        target_essay_set: int in {1, 2, ..., 8}
        lowercase: bool

    Returns:
        essays and scores from given target essay set
        essays: 1-D string list
        labels: 1-D int list
    """
    label_dict = {}
    essays = []
    labels = []
    with open(label_file, errors='ignore') as fd:
        if headline:
            fd.readline()
        lines = fd.readlines()
    for line in lines:
        prediction_id, predicted_score = line.split(',')
        label_dict[prediction_id] = int(predicted_score)
    with open(essay_file, errors='ignore') as fd:
        if headline:
            fd.readline()
        lines = fd.readlines()
    for line in lines:
        splits = line.split('\t')
        essay_set = int(splits[1])
        if target_essay_set == essay_set:
            if lowercase:
                essay = splits[2].strip('"').lower()
            else:
                essay = splits[2].strip('"')
        else:
            essay = ''
        if essay_set == 1:
            essays.append(essay)
            labels.append(label_dict[splits[3]] - 2)
    return essays, labels


def convert_essays_to_words(essays):
    """Convert essays to words, each essay string to one word list.

    Args:
        essays: 1-D string list

    Returns:
        essays_words: 2-D string list
    """
    essays_words = []
    for essay in essays:
        essays_words.append(
            regexp_tokenize(essay, pattern='\w+|[^\w\s]'))
    return essays_words


def pad_essays_words(essays_words, max_length=500):
    """Pad essays so that every essay has the same number of words.

    Args:
        essays_words: 2-D string list
        max_length: int

    Returns:
        padded_essays_words: 2-D string list
    """
    padded_essays_words = []
    for essay_words in essays_words:
        if len(essay_words) <= max_length:
            padded_essays_words.append(
                essay_words + ['PAD'] * (max_length - len(essay_words)))
        else:
            padded_essays_words.append(essay_words[:max_length])
    return padded_essays_words


def generate_words_set(essays):
    """Generate words set for given essay list.

    Args:
        essays: 1-D sentence string list

    Returns:
        a string set that contains all words from the given essays.
    """
    words_set = set()
    for essay in essays:
        words_set.update(regexp_tokenize(essay, pattern='\w+|[^\w\s]'))
    return words_set


def rebuild_word_embedding_file(words_set, word_embedding_path, out_path):
    """Rebuild word embedding file for given word set.

    Args:
        words_set: a string set
        word_embedding_path: string, big file path (GloVe)
            e.g. glove.6B.200d.txt
        out_path: string, small file path
    """
    with open(word_embedding_path, encoding='utf-8') as fd:
        lines = fd.readlines()
    with open(out_path, 'w') as fd:
        for line in lines:
            if line.split()[0] in words_set:
                fd.write(line)


def convert_essay_words_to_embeddings(
        essays, word_embedding_path, embedding_dim=200, essay_max_length=500):
    """Convert words to embeddings

    Args:
        essays: 2-D (sentence, word) string list
        word_embedding_path: string
        embedding_dim: int
        essay_max_length: int

    Returns:
        3-D (sentence, word, embedding) float list
    """
    with open(word_embedding_path) as fd:
        lines = fd.readlines()
    embedding_dict = {
        'PAD': [0.] * embedding_dim,
        'UNK': uniform(-1.0, 1.0, embedding_dim)}
    for line in lines:
        splits = line.split()
        word = splits[0]
        embeddings = [float(embedding) for embedding in splits[1:]]
        embedding_dict[word] = embeddings
    embedded_essays = []
    for essay in essays:
        essay_embedding = []
        for word in essay:
            if word in embedding_dict:
                essay_embedding.append(embedding_dict[word])
            else:
                essay_embedding.append(embedding_dict['UNK'])
        embedded_essays.append(essay_embedding)
    return reshape(embedded_essays, [-1, essay_max_length, embedding_dim])


def get_train_essay_embeddings(file_path, embedding_path):
    """Load data to embedding list.

    Args:
        file_path: string, data/training_set_rel3.tsv
        embedding_path: string, data/word_embedding_glove_6B_200d.txt

    Returns:
        essays_embedding: 3-D (essay, word, embedding)
        labels: 1-D (essay)
    """
    essays, labels = load_train_data(file_path)
    essays_words = convert_essays_to_words(essays)
    padded_essays_words = pad_essays_words(essays_words)
    essays_embedding = convert_essay_words_to_embeddings(
        padded_essays_words, embedding_path)
    labels = reshape(labels, [-1])
    return essays_embedding, labels


def get_valid_essay_embeddings(essay_file, label_file, embedding_path):
    """Load data to embedding list.

    Args:
        essay_file: string, data/valid_set.tsv
        label_file: string, data/valid_sample_submission_2_column.tsv
        embedding_path: string, data/word_embedding_glove_6B_200d.txt

    Returns:
        essays_embedding: 3-D (essay, word, embedding)
        labels: 1-D (essay)
    """

    essays, labels = load_valid_data(essay_file, label_file)
    essays_words = convert_essays_to_words(essays)
    padded_essays_words = pad_essays_words(essays_words)
    essays_embedding = convert_essay_words_to_embeddings(
        padded_essays_words, embedding_path)
    labels = reshape(labels, [-1])
    return essays_embedding, labels
