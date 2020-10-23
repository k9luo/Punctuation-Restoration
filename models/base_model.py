import numpy as np
import os
import tensorflow as tf

from abc import abstractmethod, ABC
from tensorflow.contrib.crf import viterbi_decode, crf_log_likelihood
from utils.io import load_dataset
from utils.logger import get_logger


class BaseModel(ABC):
    def __init__(self, config):
        self.cfg = config
        self._initialize_config()
        self.sess, self.saver = None, None
        self._add_placeholders()
        self._build_embedding_op()
        self._build_model_op()
        self._build_loss_op()
        self._build_train_op()
        print('params number: {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        self.initialize_session()

    def _initialize_config(self):
        # create folders and logger
        if not os.path.exists(self.cfg["checkpoint_path"]):
            os.makedirs(self.cfg["checkpoint_path"])
        if not os.path.exists(self.cfg["summary_path"]):
            os.makedirs(self.cfg["summary_path"])
        self.logger = get_logger(os.path.join(self.cfg["checkpoint_path"], "log.txt"))
        # load dictionary
        dict_data = load_dataset(self.cfg["vocab"])
        self.word_dict = dict_data["word_dict"]
        self.tag_dict = dict_data["tag_dict"]
        del dict_data
        self.word_vocab_size = len(self.word_dict)
        self.tag_vocab_size = len(self.tag_dict)
        self.rev_word_dict = dict([(idx, word) for word, idx in self.word_dict.items()])
        self.rev_tag_dict = dict([(idx, tag) for tag, idx in self.tag_dict.items()])

    def initialize_session(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver(max_to_keep=self.cfg["max_to_keep"])
        self.sess.run(tf.global_variables_initializer())

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.cfg["checkpoint_path"])  # get checkpoint state

        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_session(self, epoch):
        self.saver.save(self.sess, self.cfg["checkpoint_path"] + self.cfg["model_name"], global_step=epoch)

    def close_session(self):
        self.sess.close()

    def _add_summary(self):
        self.summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.cfg["summary_path"] + "train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.cfg["summary_path"] + "test")

    @staticmethod
    def viterbi_decode(logits, trans_params, seq_len):
        viterbi_sequences = []

        for logit, lens in zip(logits, seq_len):
            logit = logit[:lens]  # keep only the valid steps
            viterbi_seq, viterbi_score = viterbi_decode(logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences

    @abstractmethod
    def _add_placeholders(self):
        pass

    @abstractmethod
    def _get_feed_dict(self, data):
        pass

    @abstractmethod
    def _build_embedding_op(self):
        pass

    @abstractmethod
    def _build_model_op(self):
        pass

    @abstractmethod
    def _build_loss_op(self):
        pass

    def _build_train_op(self):
        with tf.variable_scope("train_step"):
            if self.cfg["optimizer"] == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
            elif self.cfg["optimizer"] == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            elif self.cfg["optimizer"] == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            elif self.cfg["optimizer"] == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
            else:  # default adam optimizer
                if self.cfg["optimizer"] != 'adam':
                    print('Unsupported optimizing method {}. Using default adam optimizer.'
                          .format(self.cfg["optimizer"]))
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        if self.cfg["grad_clip"] is not None and self.cfg["grad_clip"] > 0:
            grads, vs = zip(*optimizer.compute_gradients(self.loss))
            grads, _ = tf.clip_by_global_norm(grads, self.cfg["grad_clip"])
            self.train_op = optimizer.apply_gradients(zip(grads, vs))
        else:
            self.train_op = optimizer.minimize(self.loss)

    @abstractmethod
    def _predict_op(self, data):
        pass

    @abstractmethod
    def train_epoch(self, train_set, valid_data, epoch):
        pass

    @abstractmethod
    def train(self, train_set, valid_data, valid_set, test_set):
        pass

    @abstractmethod
    def evaluate(self, file):
        pass

    @abstractmethod
    def inference(self, sentence):
        pass
