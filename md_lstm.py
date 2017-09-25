import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear


def ln(tensor, scope=None, epsilon=1e-5):
    """ Layer normalizes a 2D tensor along its second axis """
    assert (len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    ln_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return ln_initial * scale + shift


class MultiDimensionalLSTMCell(RNNCell):
    """
    Adapted from TF's BasicLSTMCell to use Layer Normalization.
    Note that state_is_tuple is always True.
    """

    def __init__(self, num_units, forget_bias=0.0, activation=tf.nn.tanh):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM).
        @param: inputs (batch,n)
        @param state: the states and hidden unit of the two cells
        """
        with tf.variable_scope(scope or type(self).__name__):
            c1, c2, h1, h2 = state

            # change bias argument to False since LN will add bias via shift
            concat = _linear([inputs, h1, h2], 5 * self._num_units, False)

            i, j, f1, f2, o = tf.split(value=concat, num_or_size_splits=5, axis=1)

            # add layer normalization to each gate
            i = ln(i, scope='i/')
            j = ln(j, scope='j/')
            f1 = ln(f1, scope='f1/')
            f2 = ln(f2, scope='f2/')
            o = ln(o, scope='o/')

            new_c = (c1 * tf.nn.sigmoid(f1 + self._forget_bias) +
                     c2 * tf.nn.sigmoid(f2 + self._forget_bias) + tf.nn.sigmoid(i) *
                     self._activation(j))

            # add layer_normalization in calculation of new hidden state
            new_h = self._activation(ln(new_c, scope='new_h/')) * tf.nn.sigmoid(o)
            new_state = LSTMStateTuple(new_c, new_h)

            return new_h, new_state


def multi_dimensional_rnn_while_loop(rnn_size, input_data, context_wind_shape, dims=None, scope_n="layer1"):
    """Implements naive multi dimension recurrent neural networks

    @param rnn_size: the hidden units
    @param input_data: the data to process of shape [batch,h,w,channels]
    @param context_wind_shape: [height,width] of the windows
    @param dims: dimensions to reverse the input data,eg.
        dims=[False,True,True,False] => true means reverse dimension
    @param scope_n : the scope

    returns [batch,h/sh[0],w/sh[1],channels*sh[0]*sh[1]] the output of the lstm
    """
    with tf.variable_scope("MultiDimensionalLSTMCell-" + scope_n):
        cell = MultiDimensionalLSTMCell(rnn_size)

        # shape = input_data.get_shape().as_list()

        # Get the symbolic shape of the data
        # shape: (batch_size, height, width, input_dim)
        shape = tf.shape(input_data)
        batch_size, h, w, inp_dim = tf.unstack(shape)

        # Add padding if the height and width is not evenly divisible by the context window sizes
        # Symbolic pad function
        def pad(value, axis, context_size):
            shp = tf.shape(value)
            m = tf.mod(shp[axis], context_size)
            pad_amt = tf.cond(
                tf.not_equal(m, 0),
                lambda: context_size - m,
                lambda: tf.constant(0)
            )

            pad_shape = tf.unstack(shp)
            pad_shape[axis] = pad_amt

            padding = tf.zeros(shape=pad_shape)

            return tf.concat([value, padding], axis=axis)

        # Pad the height
        input_data_padded = pad(input_data, 1, context_wind_shape[0])

        # Pad the width
        input_data_padded = pad(input_data_padded, 2, context_wind_shape[1])

        # Calculate the reduced height and width dimensions to account for context windows
        h_red, w_red = tf.unstack(tf.shape(input_data_padded[0, :, :, 0]))
        h_red, w_red = h_red / context_wind_shape[0], w_red / context_wind_shape[1]

        # Recalculate the feature dimension to account for the size of context window
        context_features_size = context_wind_shape[1] * context_wind_shape[0] * input_data_padded.get_shape().as_list()[-1]

        # Reshape input data to group the features in a context window
        x = tf.reshape(input_data, [batch_size, h_red, w_red, context_features_size])

        # Perform reversing of dimensions
        if dims is not None:
            assert dims[0] is False and dims[3] is False
            x = tf.reverse(x, dims)

        # Shuffle dimensions to look like (height, width, batch_size, context_wind_feature_size)
        x = tf.transpose(x, [1, 2, 0, 3])
        x = tf.reshape(x, [-1, batch_size, context_features_size])
        # x = tf.split(axis=0, num_or_size_splits=h * w, value=x)

        # sequence_length = tf.ones(shape=(batch_size,), dtype=tf.int32) * shape[0]
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=h_red * w_red, name='input_ta')
        inputs_ta = inputs_ta.unstack(x)
        states_ta = tf.TensorArray(dtype=tf.float32, size=h_red * w_red + 1, name='state_ta', clear_after_read=False)
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=h_red * w_red, name='output_ta')

        # initial cell and hidden states
        states_ta = states_ta.write(h * w, LSTMStateTuple(tf.zeros([batch_size, rnn_size], tf.float32),
                                                          tf.zeros([batch_size, rnn_size], tf.float32)))

        # helper methods for getting the index of the above state and the left state
        def get_up(t_, w_):
            # state above is one row back from the current index
            return t_ - tf.constant(w_)

        def get_last(t_, w_):
            # state to the left is just one index back from the current index
            return t_ - tf.constant(1)

        # initialize step counters for the multi-dimensional while loop
        time = tf.constant(0)
        zero = tf.constant(0)

        # body of while loop
        def body(time_, outputs_ta_, states_ta_):

            # if not in first row, state up is one row back. otherwise it's the 0 state we have a placeholder for
            state_up = tf.cond(tf.less_equal(tf.constant(w_red), time_),
                               lambda: states_ta_.read(get_up(time_, w_red)),
                               lambda: states_ta_.read(h_red * w_red))

            # if not the first column, state is one index back. otherwise it's the 0 state we hav ea placeholderfor
            state_last = tf.cond(tf.less(zero, tf.mod(time_, tf.constant(w_red))),
                                 lambda: states_ta_.read(get_last(time_, w_red)),
                                 lambda: states_ta_.read(h_red * w_red))

            # Combine the cell states for the up and left states into a tuple
            current_state = state_up[0], state_last[0], state_up[1], state_last[1]

            # Run the multi-dimensional cell step
            out, state = cell(inputs_ta.read(time_), current_state)

            # Write the output and the cell state
            outputs_ta_ = outputs_ta_.write(time_, out)
            states_ta_ = states_ta_.write(time_, state)  # bc multi-dim, need to actually record states

            return time_ + 1, outputs_ta_, states_ta_

        # while loop conditional
        def condition(time_, outputs_ta_, states_ta_):
            return tf.less(time_, tf.constant(h_red * w_red))

        result, outputs_ta, states_ta = tf.while_loop(condition, body, [time, outputs_ta, states_ta],
                                                      parallel_iterations=1)

        # stack the outputs and states
        outputs = outputs_ta.stack()
        states = states_ta.stack()

        # reshape (do we need this?)
        y = tf.reshape(outputs, [h_red, w_red, batch_size, rnn_size])

        # put the batch back as the first dimension
        y = tf.transpose(y, [2, 0, 1, 3])

        # reverse back on dims if we had reversed originally
        if dims is not None:
            y = tf.reverse(y, dims)

        # returns the hidden outputs and the states
        return y, states
