import codecs
import numpy as np
import os
import tensorflow as tf

from evaluation.general_performance import compute_score
from models.base_model import BaseModel
from models.nns import AttentionCell, highway_network, DenselyConnectedBiRNN
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.python.ops.rnn import dynamic_rnn
from utils.data_utils import END, UNK, SPACE, PUNCTUATION_MAPPING, EOS_TOKENS, PADDING_TOKEN
from utils.logger import Progbar


class PunctuationRestorationModel(BaseModel):
    def __init__(self, config):
        super(PunctuationRestorationModel, self).__init__(config)

    def _add_placeholders(self):
        self.words = tf.placeholder(tf.int32, shape=[None, None], name="words")  # shape = (batch_size, max_time)
        self.tags = tf.placeholder(tf.int32, shape=[None, None], name="tags")  # shape = (batch_size, max_time - 1)
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name="seq_len")
        # hyper-parameters
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        self.is_train = tf.placeholder(tf.bool, shape=[], name="is_train")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.drop_rate = tf.placeholder(tf.float32, name="dropout_rate")
        self.lr = tf.placeholder(tf.float32, name="learning_rate")

    def _get_feed_dict(self, batch, keep_prob=1.0, is_train=False, lr=None):
        feed_dict = {self.words: batch["words"], self.seq_len: batch["seq_len"], self.batch_size: batch["batch_size"]}

        if "tags" in batch:
            feed_dict[self.tags] = batch["tags"]

        feed_dict[self.keep_prob] = keep_prob
        feed_dict[self.drop_rate] = 1.0 - keep_prob
        feed_dict[self.is_train] = is_train

        if lr is not None:
            feed_dict[self.lr] = lr

        return feed_dict

    def _build_embedding_op(self):
        with tf.variable_scope("embeddings"):
            if not self.cfg["use_pretrained"]:
                self.word_embeddings = tf.get_variable(name="emb", dtype=tf.float32, trainable=True,
                                                       shape=[self.word_vocab_size, self.cfg["emb_dim"]])
            else:
                word_emb_1 = tf.Variable(np.load(self.cfg["pretrained_emb"])["embeddings"], name="word_emb_1",
                                         dtype=tf.float32, trainable=self.cfg["tuning_emb"])
                word_emb_2 = tf.get_variable(name="word_emb_2", shape=[3, self.cfg["emb_dim"]], dtype=tf.float32,
                                             trainable=True)  # For UNK, NUM and END
                self.word_embeddings = tf.concat([word_emb_1, word_emb_2], axis=0)
            word_emb = tf.nn.embedding_lookup(self.word_embeddings, self.words, name="word_emb")
            print("word embedding shape: {}".format(word_emb.get_shape().as_list()))

            if self.cfg["use_highway"]:
                self.word_emb = highway_network(word_emb, self.cfg["highway_layers"], use_bias=True, bias_init=0.0,
                                                keep_prob=self.keep_prob, is_train=self.is_train)
            else:
                self.word_emb = tf.layers.dropout(word_emb, rate=self.drop_rate, training=self.is_train)

    def _build_model_op(self):
        with tf.variable_scope("densely_connected_bi_rnn"):
            dense_bi_rnn = DenselyConnectedBiRNN(self.cfg["num_layers"], self.cfg["num_units_list"],
                                                 cell_type=self.cfg["cell_type"])
            context = dense_bi_rnn(self.word_emb, seq_len=self.seq_len)
            print("densely connected bi_rnn output shape: {}".format(context.get_shape().as_list()))

        with tf.variable_scope("attention"):
            p_context = tf.layers.dense(context, units=2 * self.cfg["num_units_list"][-1], use_bias=True,
                                        bias_initializer=tf.constant_initializer(0.0))
            context = tf.transpose(context, [1, 0, 2])
            p_context = tf.transpose(p_context, [1, 0, 2])
            attn_cell = AttentionCell(self.cfg["num_units_list"][-1], context, p_context)
            attn_outs, _ = dynamic_rnn(attn_cell, context[1:, :, :], sequence_length=self.seq_len - 1, dtype=tf.float32,
                                       time_major=True)
            attn_outs = tf.transpose(attn_outs, [1, 0, 2])
            print("attention output shape: {}".format(attn_outs.get_shape().as_list()))

        with tf.variable_scope("project"):
            self.logits = tf.layers.dense(attn_outs, units=self.tag_vocab_size, use_bias=True,
                                          bias_initializer=tf.constant_initializer(0.0))
            print("logits shape: {}".format(self.logits.get_shape().as_list()))

    def _build_loss_op(self):
        if self.cfg["use_crf"]:
            crf_loss, self.trans_params = crf_log_likelihood(self.logits, self.tags, self.seq_len - 1)
            self.loss = tf.reduce_mean(-crf_loss)
        else:  # using softmax
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.tags)
            mask = tf.sequence_mask(self.seq_len)
            self.loss = tf.reduce_mean(tf.boolean_mask(losses, mask))

        if self.cfg["l2_reg"] is not None and self.cfg["l2_reg"] > 0.0:  # l2 regularization
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name])
            self.loss += self.cfg["l2_reg"] * l2_loss

        tf.summary.scalar("loss", self.loss)

    def _predict_op(self, data):
        feed_dict = self._get_feed_dict(data)
        if self.cfg["use_crf"]:
            logits, trans_params = self.sess.run([self.logits, self.trans_params], feed_dict=feed_dict)
            return self.viterbi_decode(logits, trans_params, data["seq_len"] - 1)
        else:
            pred_logits = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
            logits = self.sess.run(pred_logits, feed_dict=feed_dict)
            return logits

    def train_epoch(self, train_set, valid_data, epoch):
        num_batches = len(train_set)
        prog = Progbar(target=num_batches)
        total_cost, total_samples = 0, 0

        for i, batch in enumerate(train_set):
            feed_dict = self._get_feed_dict(batch, is_train=True, keep_prob=self.cfg["keep_prob"], lr=self.cfg["lr"])
            _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.summary], feed_dict=feed_dict)
            cur_step = (epoch - 1) * num_batches + (i + 1)
            total_cost += train_loss
            total_samples += np.array(batch["words"]).shape[0]
            prog.update(i + 1, [("Global Step", int(cur_step)), ("Train Loss", train_loss),
                                ("Perplexity", np.exp(total_cost / total_samples))])
            self.train_writer.add_summary(summary, cur_step)

            if i % 100 == 0:
                valid_feed_dict = self._get_feed_dict(valid_data)
                valid_summary = self.sess.run(self.summary, feed_dict=valid_feed_dict)
                self.test_writer.add_summary(valid_summary, cur_step)

    def train(self, train_set, valid_data, valid_text, test_texts):  # test_texts: [ref, asr]
        self.logger.info("Start training...")
        best_f1, no_imprv_epoch = -np.inf, 0
        self._add_summary()

        for epoch in range(1, self.cfg["epochs"] + 1):
            self.logger.info("Epoch {}/{}:".format(epoch, self.cfg["epochs"]))
            self.train_epoch(train_set, valid_data, epoch)

            ref_f1 = self.evaluate(test_texts[0])["F1"] * 100.0  # use ref to compute best F1
            asr_f1 = self.evaluate(test_texts[1])["F1"] * 100.0
            if ref_f1 >= best_f1:
                best_f1 = ref_f1
                no_imprv_epoch = 0
                self.save_session(epoch)
                self.logger.info(" -- new BEST score on ref dataset: {:04.2f}, on asr dataset: {:04.2f}"
                                 .format(best_f1, asr_f1))
            else:
                no_imprv_epoch += 1
                if no_imprv_epoch >= self.cfg["no_imprv_tolerance"]:
                    self.logger.info("early stop at {}th epoch without improvement, BEST score on ref dataset: {:04.2f}"
                                     .format(epoch, best_f1))
                    break

        self.train_writer.close()
        self.test_writer.close()

    def evaluate(self, file):
        save_path = os.path.join(self.cfg["checkpoint_path"], "result.txt")
        with codecs.open(file, mode="r", encoding="utf-8") as f:
            text = f.read().split()
        text = [w for w in text if w not in self.tag_dict and w not in PUNCTUATION_MAPPING] + [END]
        index = 0
        with codecs.open(save_path, mode="w", encoding="utf-8") as f_out:
            while True:
                subseq = text[index: index + self.cfg["max_sequence_len"]]
                if len(subseq) == 0:
                    break
                # create feed data
                cvrt_seq = np.array([[self.word_dict.get(w, self.word_dict[UNK]) for w in subseq]], dtype=np.int32)
                seq_len = np.array([len(v) for v in cvrt_seq], dtype=np.int32)
                data = {"words": cvrt_seq, "seq_len": seq_len, "batch_size": 1}
                # predict
                predicts = self._predict_op(data)
                # write to file
                f_out.write(subseq[0])
                last_eos_idx = 0
                punctuations = []
                for preds_t in predicts[0]:
                    punctuation = self.rev_tag_dict[preds_t]
                    punctuations.append(punctuation)
                    if punctuation in EOS_TOKENS:
                        last_eos_idx = len(punctuations)
                if subseq[-1] == END:
                    step = len(subseq) - 1
                elif last_eos_idx != 0:
                    step = last_eos_idx
                else:
                    step = len(subseq) - 1
                for j in range(step):
                    f_out.write(" " + punctuations[j] + " " if punctuations[j] != SPACE else " ")
                    if j < step - 1:
                        f_out.write(subseq[1 + j])
                if subseq[-1] == END:
                    break
                index += step
        out_str, f1, err, ser = compute_score(file, save_path)
        score = {"F1": f1, "ERR": err, "SER": ser}
        self.logger.info("\nEvaluate on {}:\n{}\n".format(file, out_str))
        try:  # delete output file after compute scores
            os.remove(save_path)
        except OSError:
            pass
        return score

    def inference(self, sentence):
        text = sentence.split()
        text = [w for w in text if w not in self.tag_dict and w not in PUNCTUATION_MAPPING]
        padding_length = 1
        text += [PADDING_TOKEN]  # To predict the last punctuation, append an extra token at the end
        text += [END]

        output = []
        index = 0

        while True:
            subseq = text[index: index + self.cfg["max_sequence_len"]]
            if len(subseq) == 0:
                break

            # create feed data
            cvrt_seq = np.array([[self.word_dict.get(w, self.word_dict[UNK]) for w in subseq]], dtype=np.int32)
            seq_len = np.array([len(v) for v in cvrt_seq], dtype=np.int32)
            data = {"words": cvrt_seq, "seq_len": seq_len, "batch_size": 1}
            # predict
            predicts = self._predict_op(data)

            output.append(subseq[0])
            last_eos_idx = 0
            punctuations = []
            for preds_t in predicts[0]:
                punctuation = self.rev_tag_dict[preds_t]
                punctuations.append(punctuation)
                if punctuation in EOS_TOKENS:
                    last_eos_idx = len(punctuations)

            if subseq[-1] == END:
                step = len(subseq) - padding_length - 1
            elif last_eos_idx != 0:
                step = last_eos_idx
            else:
                step = len(subseq) - padding_length - 1
            for j in range(step):
                output.append(" " + punctuations[j] + " " if punctuations[j] != SPACE else " ")
                if j < step - 1:
                    output.append(subseq[1 + j])
            if subseq[-1] == END:
                break
            index += step

        return "".join(output)[:-1]
