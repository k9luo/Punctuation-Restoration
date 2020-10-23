import tensorflow as tf

from functools import reduce
from operator import mul
from tensorflow.python.ops.rnn_cell import RNNCell, LSTMCell, GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn


def highway_layer(inputs, use_bias=True, bias_init=0.0, keep_prob=1.0, is_train=False, scope=None):
    with tf.variable_scope(scope or "highway_layer"):
        hidden = inputs.get_shape().as_list()[-1]
        with tf.variable_scope("trans"):
            trans = tf.layers.dropout(inputs, rate=1.0 - keep_prob, training=is_train)
            trans = tf.layers.dense(trans, units=hidden, use_bias=use_bias, bias_initializer=tf.constant_initializer(
                bias_init), activation=None)
            trans = tf.nn.relu(trans)
        with tf.variable_scope("gate"):
            gate = tf.layers.dropout(inputs, rate=1.0 - keep_prob, training=is_train)
            gate = tf.layers.dense(gate, units=hidden, use_bias=use_bias, bias_initializer=tf.constant_initializer(
                bias_init), activation=None)
            gate = tf.nn.sigmoid(gate)
        outputs = gate * trans + (1 - gate) * inputs
        return outputs


def highway_network(inputs, highway_layers=2, use_bias=True, bias_init=0.0, keep_prob=1.0, is_train=False, scope=None):
    with tf.variable_scope(scope or "highway_network"):
        prev = inputs
        cur = None
        for idx in range(highway_layers):
            cur = highway_layer(prev, use_bias, bias_init, keep_prob, is_train, scope="highway_layer_{}".format(idx))
            prev = cur
        return cur


class BiRNN:
    def __init__(self, num_units, cell_type='lstm', scope=None):
        self.cell_fw = GRUCell(num_units) if cell_type == 'gru' else LSTMCell(num_units)
        self.cell_bw = GRUCell(num_units) if cell_type == 'gru' else LSTMCell(num_units)
        self.scope = scope or "bi_rnn"

    def __call__(self, inputs, seq_len, use_last_state=False, time_major=False):
        assert not time_major, "BiRNN class cannot support time_major currently"
        with tf.variable_scope(self.scope):
            flat_inputs = flatten(inputs, keep=2)  # reshape to [-1, max_time, dim]
            seq_len = flatten(seq_len, keep=0)  # reshape to [x] (one dimension sequence)
            outputs, ((_, h_fw), (_, h_bw)) = bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, flat_inputs,
                                                                        sequence_length=seq_len, dtype=tf.float32)
            if use_last_state:  # return last states
                output = tf.concat([h_fw, h_bw], axis=-1)  # shape = [-1, 2 * num_units]
                output = reconstruct(output, ref=inputs, keep=2, remove_shape=1)  # remove the max_time shape
            else:
                output = tf.concat(outputs, axis=-1)  # shape = [-1, max_time, 2 * num_units]
                output = reconstruct(output, ref=inputs, keep=2)  # reshape to same as inputs, except the last two dim
            return output


class DenselyConnectedBiRNN:
    """Implement according to Densely Connected Bidirectional LSTM with Applications to Sentence Classification
       https://arxiv.org/pdf/1802.00889.pdf"""
    def __init__(self, num_layers, num_units, cell_type='lstm', scope=None):
        if type(num_units) == list:
            assert len(num_units) == num_layers, "if num_units is a list, then its size should equal to num_layers"
        self.dense_bi_rnn = []
        for i in range(num_layers):
            units = num_units[i] if type(num_units) == list else num_units
            self.dense_bi_rnn.append(BiRNN(units, cell_type, scope='bi_rnn_{}'.format(i)))
        self.num_layers = num_layers
        self.scope = scope or "densely_connected_bi_rnn"

    def __call__(self, inputs, seq_len, time_major=False):
        assert not time_major, "DenseConnectBiRNN class cannot support time_major currently"
        # this function does not support return_last_state method currently
        with tf.variable_scope(self.scope):
            flat_inputs = flatten(inputs, keep=2)  # reshape to [-1, max_time, dim]
            seq_len = flatten(seq_len, keep=0)  # reshape to [x] (one dimension sequence)
            cur_inputs = flat_inputs
            for i in range(self.num_layers):
                cur_outputs = self.dense_bi_rnn[i](cur_inputs, seq_len)
                if i < self.num_layers - 1:
                    cur_inputs = tf.concat([cur_inputs, cur_outputs], axis=-1)
                else:
                    cur_inputs = cur_outputs
            output = reconstruct(cur_inputs, ref=inputs, keep=2)
            return output


class AttentionCell(RNNCell):  # time_major based
    """Implement of https://pdfs.semanticscholar.org/8785/efdad2abc384d38e76a84fb96d19bbe788c1.pdf?_ga=2.156364859.18139
    40814.1518068648-1853451355.1518068648
    refer: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell_impl.py"""
    def __init__(self, num_units, memory, pmemory, cell_type='lstm'):
        super(AttentionCell, self).__init__()
        self._cell = LSTMCell(num_units) if cell_type == 'lstm' else GRUCell(num_units)
        self.num_units = num_units
        self.memory = memory
        self.pmemory = pmemory
        self.mem_units = memory.get_shape().as_list()[-1]

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def compute_output_shape(self, input_shape):
        pass

    def __call__(self, inputs, state, scope=None):
        c, m = state
        # (max_time, batch_size, att_unit)
        ha = tf.nn.tanh(tf.add(self.pmemory, tf.layers.dense(m, self.mem_units, use_bias=False, name="wah")))
        alphas = tf.squeeze(tf.exp(tf.layers.dense(ha, units=1, use_bias=False, name='way')), axis=[-1])
        alphas = tf.div(alphas, tf.reduce_sum(alphas, axis=0, keepdims=True))  # (max_time, batch_size)
        # (batch_size, att_units)
        w_context = tf.reduce_sum(tf.multiply(self.memory, tf.expand_dims(alphas, axis=-1)), axis=0)
        h, new_state = self._cell(inputs, state)
        lfc = tf.layers.dense(w_context, self.num_units, use_bias=False, name='wfc')
        # (batch_size, num_units)
        fw = tf.sigmoid(tf.layers.dense(lfc, self.num_units, use_bias=False, name='wff') +
                        tf.layers.dense(h, self.num_units, name='wfh'))
        hft = tf.multiply(lfc, fw) + h  # (batch_size, num_units)
        return hft, new_state


def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep, remove_shape=None):
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    if remove_shape is not None:
        tensor_start = tensor_start + remove_shape
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out
