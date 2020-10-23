import codecs
import numpy as np
import os

from collections import Counter
from tqdm import tqdm
from utils.data_utils import SPACE, UNK, NUM, END, is_number, CRAP_TOKENS, PUNCTUATION_VOCABULARY, PUNCTUATION_MAPPING, glove_sizes, EOS_TOKENS
from utils.io import write_json


def build_vocab_list(data_files, min_word_count, max_vocab_size):
    """
    Build a vocabulary for words.
    Args:
        data_files: The dataset that vocabularies are built from.
        min_word_count: Minimal word count can be considered in the word vocabulary.
        max_vocab_size: Maximal vocabulary size.
    Returns: A list of tuple of words and their frequencies in the word vocabulary.
    """
    word_counter = Counter()

    for file in data_files:
        with codecs.open(file, mode="r", encoding="utf-8") as f:
            for line in f:
                for word in line.lstrip().rstrip().split():
                    if word in CRAP_TOKENS or word in PUNCTUATION_VOCABULARY or word in PUNCTUATION_MAPPING:
                        continue

                    if is_number(word):
                        word_counter[NUM] += 1
                        continue

                    word_counter[word] += 1

    word_vocab = [(word, count) for word, count in word_counter.most_common() if
                  count >= min_word_count and word != UNK and
                  word != NUM][:max_vocab_size]
    return word_vocab


def build_vocabulary(word_vocab):
    """
    Add number, end, unknown token in word vocabulary.
    Args:
        word_vocab: A list of words in word vocabulary.
    Returns: Word dictionary with word as key and its index in vocabulary as index.
    """
    if NUM not in word_vocab:
        word_vocab.append(NUM)
    if END not in word_vocab:
        word_vocab.append(END)
    if UNK not in word_vocab:
        word_vocab.append(UNK)
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    return word_dict


def load_glove_vocab(glove_path, glove_name):
    """
    Load GloVe vocabulary.
    Args:
        glove_path: The path to GloVe vocabulary.
        glove_name: The path to GloVe vocabulary.
    Returns: A set of words in GloVe vocabulary.
    """
    vocab = set()
    total = glove_sizes[glove_name]
    with codecs.open(glove_path, mode='r', encoding='utf-8') as f:
        for line in tqdm(f, total=total, desc="Load glove vocabulary"):
            line = line.lstrip().rstrip().split(" ")
            vocab.add(line[0])
    return vocab


def filter_glove_emb(word_dict, glove_path, glove_name, dim):
    """
    Filer out words that are in word vocabulary but not in GloVe.
    Args:
        word_dict: A dictionary of words in word vocabulary.
        glove_path: The path to GloVe vocabulary.
        glove_name: The name of GloVe vocabulary.
        dim: Embedding dimension for input words/tokens.
    Returns: Word vectors for words in word vocabulary.
    """
    scale = np.sqrt(3.0 / dim)
    vectors = np.random.uniform(-scale, scale, [len(word_dict), dim])
    mask = np.zeros([len(word_dict)])
    with codecs.open(glove_path, mode='r', encoding='utf-8') as f:
        for line in tqdm(f, total=glove_sizes[glove_name], desc="Filter glove embeddings"):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = [float(x) for x in line[1:]]
            if word in word_dict:
                word_idx = word_dict[word]
                mask[word_idx] = 1
                vectors[word_idx] = np.asarray(vector)
            # since tokens in train sets are lowercase
            elif word.lower() in word_dict and mask[word_dict[word.lower()]] == 0:
                word = word.lower()
                word_idx = word_dict[word]
                mask[word_idx] = 1
                vectors[word_idx] = np.asarray(vector)
    return vectors


def build_dataset(data_files, word_dict, punct_dict, max_sequence_len):
    """
    Build datasets for downstream jobs.
    Data will consist of two sets of aligned sub-sequences (words and punctuations) of MAX_SEQUENCE_LEN tokens
    (actually punctuation sequence will be 1 element shorter).
    If a sentence is cut, then it will be added to next subsequence entirely
    (words before the cut belong to both sequences)
    Args:
        data_files: Dataset needs to be built.
        word_dict: A dictionary for words and their indexes.
        punct_dict: A dictionary for punctuations and their indexes.
        max_sequence_len: Maximal sequence length allowed.
    Returns: Dataset can be used for training, development and test. Punctuation counter can be used for exploratory
    data analysis.
    """
    punct_counter = Counter()
    dataset = []
    current_words, current_punctuations = [], []
    last_eos_idx = 0  # if it's still 0 when MAX_SEQUENCE_LEN is reached, then the sentence is too long and skipped.
    last_token_was_punctuation = True  # skip first token if it's punctuation
    # if a sentence does not fit into subsequence, then we need to skip tokens until we find a new sentence
    skip_until_eos = False

    for file in data_files:
        with codecs.open(file, 'r', encoding='utf-8') as f:
            for line in f:
                for token in line.split():
                    # First map oov punctuations to known punctuations
                    if token in PUNCTUATION_MAPPING:
                        token = PUNCTUATION_MAPPING[token]

                    if skip_until_eos:
                        if token in EOS_TOKENS:
                            skip_until_eos = False
                        continue
                    elif token in CRAP_TOKENS:
                        continue
                    elif token in punct_dict:
                        # if we encounter sequences like: "... <exclamationmark> <period> ...",
                        # then we only use the first punctuation and skip the ones that follow
                        if last_token_was_punctuation:
                            continue
                        if token in EOS_TOKENS:
                            last_eos_idx = len(current_punctuations)  # no -1, because the token is not added yet
                        punctuation = punct_dict[token]
                        current_punctuations.append(punctuation)
                        last_token_was_punctuation = True
                        punct_counter[token] += 1
                    else:
                        if not last_token_was_punctuation:
                            current_punctuations.append(punct_dict[SPACE])
                        if is_number(token):
                            token = NUM
                        word = word_dict.get(token, word_dict[UNK])
                        current_words.append(word)
                        last_token_was_punctuation = False

                    if len(current_words) == max_sequence_len:  # this also means, that last token was a word
                        assert len(current_words) == len(current_punctuations) + 1, \
                            "#words: %d; #punctuations: %d" % (len(current_words), len(current_punctuations))

                        # Sentence did not fit into subsequence - skip it
                        if last_eos_idx == 0:
                            skip_until_eos = True
                            current_words = []
                            current_punctuations = []
                            # next sequence starts with a new sentence, so is preceded by eos which is punctuation
                            last_token_was_punctuation = True
                        else:
                            subsequence = {"words": current_words[:-1] + [word_dict[END]], "tags": current_punctuations}
                            dataset.append(subsequence)
                            # Carry unfinished sentence to next subsequence
                            current_words = current_words[last_eos_idx + 1:]
                            current_punctuations = current_punctuations[last_eos_idx + 1:]
                        last_eos_idx = 0  # sequence always starts with a new sentence
    return dataset, punct_counter


def get_word_counter(word_vocab):
    """
    Convert a list of tuple of words and their frequencies in word vocabulary to dictionary.
    Key is word and value is its frequency.
    Args:
        word_vocab: A list of tuple of words and their frequencies.
    Returns: A dictionary with word as the key and its frequency in word vocabulary as the value.
    """
    return {word: count for word, count in word_vocab}


def process_data(config):
    """
    Import and preprocess raw datasets. Then export processed datasets, vocabularies, word counter, punctuation counter
    for downstream jobs or exploratory data analysis.
    Args:
        config: a dictionary contains parameters for datasets.
    Returns: None.
    """
    train_file = os.path.join(config["raw_path"], "train.txt")
    dev_file = os.path.join(config["raw_path"], "dev.txt")
    ref_file = os.path.join(config["raw_path"], "ref.txt")
    asr_file = os.path.join(config["raw_path"], "asr.txt")

    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])

    # build vocabulary
    train_word_vocab = build_vocab_list([train_file], config["min_word_count"], config["max_vocab_size"])
    train_word_counter = get_word_counter(train_word_vocab)
    train_word_vocab = list(train_word_counter.keys())

    dev_word_vocab = build_vocab_list([dev_file], config["min_word_count"], config["max_vocab_size"])
    dev_word_counter = get_word_counter(dev_word_vocab)

    ref_word_vocab = build_vocab_list([ref_file], config["min_word_count"], config["max_vocab_size"])
    ref_word_counter = get_word_counter(ref_word_vocab)

    asr_word_vocab = build_vocab_list([asr_file], config["min_word_count"], config["max_vocab_size"])
    asr_word_counter = get_word_counter(asr_word_vocab)

    if not config["use_pretrained"]:
        word_dict = build_vocabulary(train_word_vocab)
    else:
        glove_path = config["glove_path"].format(config["glove_name"], config["emb_dim"])
        glove_vocab = load_glove_vocab(glove_path, config["glove_name"])
        glove_vocab = glove_vocab & {word.lower() for word in glove_vocab}
        filtered_train_word_vocab = [word for word in train_word_vocab if word in glove_vocab]
        word_dict = build_vocabulary(filtered_train_word_vocab)
        tmp_word_dict = word_dict.copy()
        del tmp_word_dict[UNK], tmp_word_dict[NUM], tmp_word_dict[END]
        vectors = filter_glove_emb(tmp_word_dict, glove_path, config["glove_name"], config["emb_dim"])
        np.savez_compressed(config["pretrained_emb"], embeddings=vectors)

    # create indices dataset
    punct_dict = dict([(punct, idx) for idx, punct in enumerate(PUNCTUATION_VOCABULARY)])
    train_set, train_punct_counter = build_dataset([train_file], word_dict, punct_dict, config["max_sequence_len"])
    dev_set, dev_punct_counter = build_dataset([dev_file], word_dict, punct_dict, config["max_sequence_len"])
    ref_set, ref_punct_counter = build_dataset([ref_file], word_dict, punct_dict, config["max_sequence_len"])
    asr_set, asr_punct_counter = build_dataset([asr_file], word_dict, punct_dict, config["max_sequence_len"])
    vocab = {"word_dict": word_dict, "tag_dict": punct_dict}

    # write to file
    write_json(config["vocab"], vocab)
    write_json(config["train_word_counter"], train_word_counter)
    write_json(config["dev_word_counter"], dev_word_counter)
    write_json(config["ref_word_counter"], ref_word_counter)
    write_json(config["asr_word_counter"], asr_word_counter)
    write_json(config["train_punct_counter"], train_punct_counter)
    write_json(config["dev_punct_counter"], dev_punct_counter)
    write_json(config["ref_punct_counter"], ref_punct_counter)
    write_json(config["asr_punct_counter"], asr_punct_counter)
    write_json(config["train_set"], train_set)
    write_json(config["dev_set"], dev_set)
    write_json(config["ref_set"], ref_set)
    write_json(config["asr_set"], asr_set)
