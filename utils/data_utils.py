import random
import re
import time

from utils.io import load_dataset


PAD = "<PAD>"
UNK = "<UNK>"
NUM = "<NUM>"
END = "</S>"
SPACE = "_SPACE"
PADDING_TOKEN = "."

# pre-set number of records in different glove embeddings
glove_sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}

# Comma, period & question mark only:
PUNCTUATION_VOCABULARY = [SPACE, "<comma>", "<period>", "<questionmark>"]
PUNCTUATION_MAPPING = {"<exclamationmark>": "<period>", "<colon>": "<comma>", "<semicolon>": "<period>",
                       "<dash>": "<comma>"}

EOS_TOKENS = {"<period>", "<questionmark>", "<exclamationmark>"}

# punctuations that are not included in vocabulary nor mapping, must be added to CRAP_TOKENS
CRAP_TOKENS = {"<doc>", "<doc.>"}


def is_number(word):
    numbers = re.compile(r"\d")
    return len(numbers.sub("", word)) / len(word) < 0.6


def pad_sequences(sequences, pad_tok=None, max_length=None):
    """
    Pad sequence.
    Args:
        sequences: A sequence needs to be padded.
        pad_tok: Token used for padding.
        max_length: Maximal length of the sequence
    Returns: The sequence after padding and the length of the padded sequence.
    """
    if pad_tok is None:
        # 0: "PAD" for words, "O" for tags
        pad_tok = 0
    if max_length is None:
        max_length = max([len(seq) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_length))
    return sequence_padded, sequence_length


def process_batch_data(batch_words, batch_tags=None):
    """
    Padding batched dataset.
    Args:
        batch_words: Words in a batch.
        batch_tags: Punctuations in a batch.
    Returns: Words and punctuations after padding.
    """
    b_words, b_words_len = pad_sequences(batch_words)
    if batch_tags is None:
        return {"words": b_words, "seq_len": b_words_len, "batch_size": len(b_words)}
    else:
        b_tags, _ = pad_sequences(batch_tags)
        return {"words": b_words, "tags": b_tags, "seq_len": b_words_len, "batch_size": len(b_words)}


def dataset_batch_iter(dataset, batch_size):
    """
    Split dataset to each batch.
    Args:
        dataset: The dataset need to be split to batches.
        batch_size: Batch size.
    Returns: A batch of dataset based on the batch size.
    """
    batch_words, batch_tags = [], []
    for record in dataset:
        batch_words.append(record["words"])
        batch_tags.append(record["tags"])
        if len(batch_words) == batch_size:
            yield process_batch_data(batch_words, batch_tags)
            batch_words, batch_tags = [], []
    if len(batch_words) > 0:
        yield process_batch_data(batch_words, batch_tags)


def split_to_batches(data, batch_size=None, shuffle=True):
    """
    Create batches of dataset.
    Args:
        data: The dataset need to be split to batches.
        batch_size: Batch size.
        shuffle: A Flag used to determine if the dataset would shuffled before split.
    Returns: Dataset that is split into batches.
    """
    if type(data) == str:
        dataset = load_dataset(data)
    else:
        dataset = data

    if shuffle:
        random.shuffle(dataset)

    batches = []
    if batch_size is None:
        for batch in dataset_batch_iter(dataset, len(dataset)):
            batches.append(batch)
        return batches[0]
    else:
        for batch in dataset_batch_iter(dataset, batch_size):
            batches.append(batch)
        return batches


def inhour(elapsed):
    return time.strftime('%H:%M:%S', time.gmtime(elapsed))
