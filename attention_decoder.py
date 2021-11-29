from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

def attention_decoder_fn_train(encoder_state,
                                                             attention_keys,
                                                             attention_values,
                                                             attention_score_fn,
                                                             attention_construct_fn,
                                                             triple_num,
                                                             output_alignments=False,
                                                             max_length=None,
                                                             name=None):
    """Attentional decoder function for `dynamic_rnn_decoder` during training.

    The `attention_decoder_fn_train` is a training function for an
    attention-based sequence-to-sequence model. It should be used when
    `dynamic_rnn_decoder` is in the training mode.

    The `attention_decoder_fn_train` is called with a set of the user arguments
    and returns the `decoder_fn`, which can be passed to the
    `dynamic_rnn_decoder`, such that

    ```
    dynamic_fn_train = attention_decoder_fn_train(encoder_state)
    outputs_train, state_train = dynamic_rnn_decoder(
            decoder_fn=dynamic_fn_train, ...)
    ```

    Further usage can be found in the `kernel_tests/seq2seq_test.py`.

    Args:
        encoder_state: The encoded state to initialize the `dynamic_rnn_decoder`.
        attention_keys: to be compared with target states.
        attention_values: to be used to construct context vectors.
        attention_score_fn: to compute similarity between key and target states.
        attention_construct_fn: to build attention states.
        name: (default: `None`) NameScope for the decoder function;
            defaults to "simple_decoder_fn_train"

    Returns:
        A decoder function with the required interface of `dynamic_rnn_decoder`
        intended for training.
    """
    with ops.name_scope(name, "attention_decoder_fn_train", [
            encoder_state, attention_keys, attention_values, attention_score_fn,
            attention_construct_fn
    ]):
        pass

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        """Decoder function used in the `dynamic_rnn_decoder` for training.

        Args:
            time: positive integer constant reflecting the current timestep.
            cell_state: state of RNNCell.
            cell_input: input provided by `dynamic_rnn_decoder`.
            cell_output: output of RNNCell.
            context_state: context state provided by `dynamic_rnn_decoder`.

        Returns:
            A tuple (done, next state, next input, emit output, next context state)
            where:

            done: `None`, which is used by the `dynamic_rnn_decoder` to indicate
            that `sequence_lengths` in `dynamic_rnn_decoder` should be used.

            next state: `cell_state`, this decoder function does not modify the
            given state.

            next input: `cell_input`, this decoder function does not modify the
            given input. The input could be modified when applying e.g. attention.

            emit output: `cell_output`, this decoder function does not modify the
            given output.

            next context state: `context_state`, this decoder function does not
            modify the given context state. The context state could be modified when
            applying e.g. beam search.
        """
        with ops.name_scope(
                name, "attention_decoder_fn_train",
                [time, cell_state, cell_input, cell_output, context_state]):
            if cell_state is None:    # first call, return encoder_state
                cell_state = encoder_state

                # init attention
                attention = _init_attention(encoder_state)
                alignments_ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name="alignments_ta", size=max_length, dynamic_size=True, infer_shape=False)
                context_state = alignments_ta
            else:
                # construct attention
                alignments_ta = context_state
                attention = attention_construct_fn(cell_output, attention_keys,
                                                                                     attention_values)
                attention, alignments = attention
                context_state = alignments_ta.write(time-1, alignments)
                cell_output = attention

            # combine cell_input and attention
            next_input = array_ops.concat([cell_input, attention], 1)

            return (None, cell_state, next_input, cell_output, context_state)

    return decoder_fn


def attention_decoder_fn_inference(output_fn,
                                                                     encoder_state,
                                                                     attention_keys,
                                                                     attention_values,
                                                                     attention_score_fn,
                                                                     attention_construct_fn,
                                                                     embeddings,
                                                                     start_of_sequence_id,
                                                                     end_of_sequence_id,
                                                                     maximum_length,
                                                                     num_decoder_symbols,
                                                                     imem=None,
                                                                     selector_fn=None,
                                                                     dtype=dtypes.int32,
                                                                     name=None):
    """Attentional decoder function for `dynamic_rnn_decoder` during inference.

    The `attention_decoder_fn_inference` is a simple inference function for a
    sequence-to-sequence model. It should be used when `dynamic_rnn_decoder` is
    in the inference mode.

    The `attention_decoder_fn_inference` is called with user arguments
    and returns the `decoder_fn`, which can be passed to the
    `dynamic_rnn_decoder`, such that

    ```
    dynamic_fn_inference = attention_decoder_fn_inference(...)
    outputs_inference, state_inference = dynamic_rnn_decoder(
            decoder_fn=dynamic_fn_inference, ...)
    ```

    Further usage can be found in the `kernel_tests/seq2seq_test.py`.

    Args:
        output_fn: An output function to project your `cell_output` onto class
        logits.

        An example of an output function;

        ```
            tf.variable_scope("decoder") as varscope
                output_fn = lambda x: layers.linear(x, num_decoder_symbols,
                                                                                        scope=varscope)

                outputs_train, state_train = seq2seq.dynamic_rnn_decoder(...)
                logits_train = output_fn(outputs_train)

                varscope.reuse_variables()
                logits_inference, state_inference = seq2seq.dynamic_rnn_decoder(
                        output_fn=output_fn, ...)
        ```

        If `None` is supplied it will act as an identity function, which
        might be wanted when using the RNNCell `OutputProjectionWrapper`.

        encoder_state: The encoded state to initialize the `dynamic_rnn_decoder`.
        attention_keys: to be compared with target states.
        attention_values: to be used to construct context vectors.
        attention_score_fn: to compute similarity between key and target states.
        attention_construct_fn: to build attention states.
        embeddings: The embeddings matrix used for the decoder sized
        `[num_decoder_symbols, embedding_size]`.
        start_of_sequence_id: The start of sequence ID in the decoder embeddings.
        end_of_sequence_id: The end of sequence ID in the decoder embeddings.
        maximum_length: The maximum allowed of time steps to decode.
        num_decoder_symbols: The number of classes to decode at each time step.
        dtype: (default: `dtypes.int32`) The default data type to use when
        handling integer objects.
        name: (default: `None`) NameScope for the decoder function;
            defaults to "attention_decoder_fn_inference"

    Returns:
        A decoder function with the required interface of `dynamic_rnn_decoder`
        intended for inference.
    """
    with ops.name_scope(name, "attention_decoder_fn_inference", [
            output_fn, encoder_state, attention_keys, attention_values,
            attention_score_fn, attention_construct_fn, embeddings,
            start_of_sequence_id, end_of_sequence_id, maximum_length,
            num_decoder_symbols, dtype
    ]):
        start_of_sequence_id = ops.convert_to_tensor(start_of_sequence_id, dtype)
        end_of_sequence_id = ops.convert_to_tensor(end_of_sequence_id, dtype)
        maximum_length = ops.convert_to_tensor(maximum_length, dtype)
        num_decoder_symbols = ops.convert_to_tensor(num_decoder_symbols, dtype)
        encoder_info = nest.flatten(encoder_state)[0]
        batch_size = encoder_info.get_shape()[0].value
        if output_fn is None:
            output_fn = lambda x: x
        if batch_size is None:
            batch_size = array_ops.shape(encoder_info)[0]

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        """Decoder function used in the `dynamic_rnn_decoder` for inference.

        The main difference between this decoder function and the `decoder_fn` in
        `attention_decoder_fn_train` is how `next_cell_input` is calculated. In
        decoder function we calculate the next input by applying an argmax across
        the feature dimension of the output from the decoder. This is a
        greedy-search approach. (Bahdanau et al., 2014) & (Sutskever et al., 2014)
        use beam-search instead.

        Args:
            time: positive integer constant reflecting the current timestep.
            cell_state: state of RNNCell.
            cell_input: input provided by `dynamic_rnn_decoder`.
            cell_output: output of RNNCell.
            context_state: context state provided by `dynamic_rnn_decoder`.

        Returns:
            A tuple (done, next state, next input, emit output, next context state)
            where:

            done: A boolean vector to indicate which sentences has reached a
            `end_of_sequence_id`. This is used for early stopping by the
            `dynamic_rnn_decoder`. When `time>=maximum_length` a boolean vector with
            all elements as `true` is returned.

            next state: `cell_state`, this decoder function does not modify the
            given state.

            next input: The embedding from argmax of the `cell_output` is used as
            `next_input`.

            emit output: If `output_fn is None` the supplied `cell_output` is
            returned, else the `output_fn` is used to update the `cell_output`
            before calculating `next_input` and returning `cell_output`.

            next context state: `context_state`, this decoder function does not
            modify the given context state. The context state could be modified when
            applying e.g. beam search.

        Raises:
            ValueError: if cell_input is not None.

        """
        with ops.name_scope(
                name, "attention_decoder_fn_inference",
                [time, cell_state, cell_input, cell_output, context_state]):
            if cell_input is not None:
                raise ValueError("Expected cell_input to be None, but saw: %s" %
                                                 cell_input)

            if cell_output is None:
                # invariant that this is time == 0
                next_input_id = array_ops.ones(
                        [batch_size,], dtype=dtype) * (start_of_sequence_id)
                done = array_ops.zeros([batch_size,], dtype=dtypes.bool)
                cell_state = encoder_state
                cell_output = array_ops.zeros(
                        [num_decoder_symbols], dtype=dtypes.float32)
                cell_input = array_ops.gather(embeddings, next_input_id)

                # init attention
                attention = _init_attention(encoder_state)
                if imem is not None:
                    coverage = tf.ones([tf.shape(encoder_state)[1], tf.shape(imem)[1]])
                    context_state = (coverage, tensor_array_ops.TensorArray(dtype=dtypes.int32, tensor_array_name="output_ids_ta", size=maximum_length, dynamic_size=True, infer_shape=False))
            else:
                # construct attention
                coverage, outputs_ta = context_state
                attention = attention_construct_fn(cell_output, attention_keys,
                                                                                     attention_values)
                if type(attention) is tuple:
                    attention, alignment = attention
                    cell_output = attention
                    alignment = tf.reshape(alignment, [batch_size, -1]) * coverage
                    #alignment = tf.reshape(alignment, [batch_size, -1])# * coverage
                    coverage_mask0 = 1.0 - tf.cast(tf.equal(alignment, tf.reduce_max(alignment, 1, keepdims=True)), tf.float32)
                    coverage_mask1 = tf.ones_like(alignment)
                    #cell_output = output_fn(cell_output)    # logits
                    #next_input_id = math_ops.cast(
                    #        math_ops.argmax(cell_output, 1), dtype=dtype)
                    #done = math_ops.equal(next_input_id, end_of_sequence_id)
                    #cell_input = array_ops.gather(embeddings, next_input_id)
                    selector = selector_fn(cell_output)
                    logit = output_fn(cell_output)
                    word_prob = nn_ops.softmax(logit) * (1 - selector)
                    #word_prob = nn_ops.softmax(logit)[:, 2:] * (1 - selector)
                    entity_prob = selector * alignment
                    mask = array_ops.reshape(math_ops.cast(math_ops.greater(tf.reduce_max(word_prob, 1), tf.reduce_max(entity_prob, 1)), dtype=dtypes.float32), [-1,1])
                    #mask = array_ops.reshape(math_ops.cast(math_ops.greater(0.7, selector), dtype=dtypes.float32), [-1,1])
                    #coverage_mask = coverage_mask1
                    coverage_mask = mask * coverage_mask1 + (1 - mask) * coverage_mask0
                    cell_input = mask * array_ops.gather(embeddings, math_ops.cast(math_ops.argmax(word_prob, 1), dtype=dtype)) + (1 - mask) * array_ops.gather_nd(imem, array_ops.concat([array_ops.reshape(math_ops.range(batch_size, dtype=dtype), [-1,1]), array_ops.reshape(math_ops.cast(math_ops.argmax(entity_prob, 1), dtype=dtype), [-1,1])], axis=1))
                    mask = array_ops.reshape(math_ops.cast(mask, dtype=dtype), [-1])
                    input_id = mask * math_ops.cast(math_ops.argmax(word_prob, 1), dtype=dtype) + (mask - 1) * math_ops.cast(math_ops.argmax(entity_prob, 1), dtype=dtype)
                    context_state = (coverage * coverage_mask, outputs_ta.write(time-1, input_id))
                    done = array_ops.reshape(math_ops.equal(input_id, end_of_sequence_id), [-1])
                    #done = tf.Print(done, ['selector', selector, 'mask', mask], summarize=1e6)
                    cell_output = logit

                else:
                    cell_output = attention

                    # argmax decoder
                    cell_output = output_fn(cell_output)    # logits
                    next_input_id = math_ops.cast(
                            math_ops.argmax(cell_output, 1), dtype=dtype)
                    done = math_ops.equal(next_input_id, end_of_sequence_id)
                    cell_input = array_ops.gather(embeddings, next_input_id)

            # combine cell_input and attention
            next_input = array_ops.concat([cell_input, attention], 1)

            # if time > maxlen, return all true vector
            done = control_flow_ops.cond(
                    math_ops.greater(time, maximum_length),
                    lambda: array_ops.ones([batch_size,], dtype=dtypes.bool),
                    lambda: done)
            return (done, cell_state, next_input, cell_output, context_state)

    return decoder_fn

def attention_decoder_fn_sample(output_fn,
                                 encoder_state,
                                 attention_keys,
                                 attention_values,
                                 attention_score_fn,
                                 attention_construct_fn,
                                 embeddings,
                                 start_of_sequence_id,
                                 end_of_sequence_id,
                                 maximum_length,
                                 num_decoder_symbols,
                                 dtype=dtypes.int32,
                                 temperature=0.8,
                                 k=0,
                                 p=0.6,
                                 name=None):
    """Attentional decoder function for `dynamic_rnn_decoder` during inference.

    The `attention_decoder_fn_inference` is a simple inference function for a
    sequence-to-sequence model. It should be used when `dynamic_rnn_decoder` is
    in the inference mode.

    The `attention_decoder_fn_inference` is called with user arguments
    and returns the `decoder_fn`, which can be passed to the
    `dynamic_rnn_decoder`, such that

    ```
    dynamic_fn_inference = attention_decoder_fn_inference(...)
    outputs_inference, state_inference = dynamic_rnn_decoder(
            decoder_fn=dynamic_fn_inference, ...)
    ```

    Further usage can be found in the `kernel_tests/seq2seq_test.py`.

    Args:
        output_fn: An output function to project your `cell_output` onto class
        logits.

        An example of an output function;

        ```
            tf.variable_scope("decoder") as varscope
                output_fn = lambda x: layers.linear(x, num_decoder_symbols,
                                                                                        scope=varscope)

                outputs_train, state_train = seq2seq.dynamic_rnn_decoder(...)
                logits_train = output_fn(outputs_train)

                varscope.reuse_variables()
                logits_inference, state_inference = seq2seq.dynamic_rnn_decoder(
                        output_fn=output_fn, ...)
        ```

        If `None` is supplied it will act as an identity function, which
        might be wanted when using the RNNCell `OutputProjectionWrapper`.

        encoder_state: The encoded state to initialize the `dynamic_rnn_decoder`.
        attention_keys: to be compared with target states.
        attention_values: to be used to construct context vectors.
        attention_score_fn: to compute similarity between key and target states.
        attention_construct_fn: to build attention states.
        embeddings: The embeddings matrix used for the decoder sized
        `[num_decoder_symbols, embedding_size]`.
        start_of_sequence_id: The start of sequence ID in the decoder embeddings.
        end_of_sequence_id: The end of sequence ID in the decoder embeddings.
        maximum_length: The maximum allowed of time steps to decode.
        num_decoder_symbols: The number of classes to decode at each time step.
        dtype: (default: `dtypes.int32`) The default data type to use when
        handling integer objects.
        name: (default: `None`) NameScope for the decoder function;
            defaults to "attention_decoder_fn_inference"

    Returns:
        A decoder function with the required interface of `dynamic_rnn_decoder`
        intended for inference.
    """
    with ops.name_scope(name, "attention_decoder_fn_inference", [
            output_fn, encoder_state, attention_keys, attention_values,
            attention_score_fn, attention_construct_fn, embeddings,
            start_of_sequence_id, end_of_sequence_id, maximum_length,
            num_decoder_symbols, dtype
    ]):
        start_of_sequence_id = ops.convert_to_tensor(start_of_sequence_id, dtype)
        end_of_sequence_id = ops.convert_to_tensor(end_of_sequence_id, dtype)
        maximum_length = ops.convert_to_tensor(maximum_length, dtype)
        num_decoder_symbols = ops.convert_to_tensor(num_decoder_symbols, dtype)
        encoder_info = nest.flatten(encoder_state)[0]
        batch_size = encoder_info.get_shape()[0].value
        if output_fn is None:
            output_fn = lambda x: x
        if batch_size is None:
            batch_size = array_ops.shape(encoder_info)[0]

    def top_k_sampling(logits, k=25):
        'k must be greater than 0'

        values, _ = tf.nn.top_k(logits, k=k)
        min_value = tf.reduce_min(values, axis=-1, keep_dims=True)
        logits = tf.where(
        logits < min_value,
        tf.ones_like(logits, dtype=logits.dtype) * -1e12,
        logits)

        #sample = tf.multinomial(tf.expand_dims(logits, 0), num_samples=1, output_dtype=tf.int32)
        sample = tf.multinomial(logits, num_samples=1)
        return sample


    def nucleus_sampling(logits, p=0.9):
        sorted_logits, sorted_indices = tf.nn.top_k(logits, k=tf.shape(logits)[1])
        sorted_indices = sorted_indices + tf.expand_dims(tf.range(tf.shape(logits)[0]) * tf.shape(logits)[1], axis=-1)
        cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits), axis=-1)
        t_sorted_indices_to_remove = cumulative_probs > p
        ''' Shift the indices to the right to keep also the first token above the threshold '''
        sorted_indices_to_remove = tf.concat([tf.zeros([tf.shape(logits)[0], 1], dtype=tf.bool), t_sorted_indices_to_remove[:,:-1]], axis=1)
        #indices = tf.range(1, tf.shape(logits)[-1], 1)
        #sorted_indices_to_remove = tf.scatter_nd(tf.expand_dims(indices, 1), t_sorted_indices_to_remove[:-1], logits.shape)
        indices_to_remove = tf.cast(tf.boolean_mask(tf.reshape(sorted_indices, [-1]), tf.reshape(sorted_indices_to_remove, [-1])), tf.int32)
        t = tf.ones([tf.shape(indices_to_remove)[0]], dtype=tf.int32)
        to_remove = tf.scatter_nd(tf.expand_dims(indices_to_remove, 1), t, [tf.shape(logits)[0] * tf.shape(logits)[1]])
        to_remove = tf.reshape(tf.cast(to_remove, tf.bool), tf.shape(logits))
        logits = tf.where(
            to_remove,
            tf.ones_like(logits, dtype=logits.dtype) * -1e12,
            logits
        )

        sample = tf.multinomial(logits, num_samples=1)
        return sample

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        """Decoder function used in the `dynamic_rnn_decoder` for inference.

        The main difference between this decoder function and the `decoder_fn` in
        `attention_decoder_fn_train` is how `next_cell_input` is calculated. In
        decoder function we calculate the next input by applying an argmax across
        the feature dimension of the output from the decoder. This is a
        greedy-search approach. (Bahdanau et al., 2014) & (Sutskever et al., 2014)
        use beam-search instead.

        Args:
            time: positive integer constant reflecting the current timestep.
            cell_state: state of RNNCell.
            cell_input: input provided by `dynamic_rnn_decoder`.
            cell_output: output of RNNCell.
            context_state: context state provided by `dynamic_rnn_decoder`.

        Returns:
            A tuple (done, next state, next input, emit output, next context state)
            where:

            done: A boolean vector to indicate which sentences has reached a
            `end_of_sequence_id`. This is used for early stopping by the
            `dynamic_rnn_decoder`. When `time>=maximum_length` a boolean vector with
            all elements as `true` is returned.

            next state: `cell_state`, this decoder function does not modify the
            given state.

            next input: The embedding from argmax of the `cell_output` is used as
            `next_input`.

            emit output: If `output_fn is None` the supplied `cell_output` is
            returned, else the `output_fn` is used to update the `cell_output`
            before calculating `next_input` and returning `cell_output`.

            next context state: `context_state`, this decoder function does not
            modify the given context state. The context state could be modified when
            applying e.g. beam search.

        Raises:
            ValueError: if cell_input is not None.

        """

        with ops.name_scope(
                name, "attention_decoder_fn_sample_inference",
                [time, cell_state, cell_input, cell_output, context_state]):
            if cell_input is not None:
                raise ValueError("Expected cell_input to be None, but saw: %s" %
                                                 cell_input)
            if cell_output is None:
                # invariant that this is time == 0
                next_input_id = array_ops.ones(
                        [batch_size,], dtype=dtype) * (start_of_sequence_id)
                done = array_ops.zeros([batch_size,], dtype=dtypes.bool)
                cell_state = encoder_state
                cell_output = array_ops.zeros(
                        [num_decoder_symbols], dtype=dtypes.float32)
                cell_input = array_ops.gather(embeddings, next_input_id)

                # init attention
                attention = _init_attention(encoder_state)
                context_state = tensor_array_ops.TensorArray(dtype=dtypes.int64, tensor_array_name='sample_index', size=maximum_length+1, dynamic_size=True, infer_shape=False)
            else:
                # construct attention
                attention = attention_construct_fn(cell_output, attention_keys,
                                                                                     attention_values)
                cell_output = attention

                # argmax decoder
                cell_output = output_fn(cell_output) / temperature   # logits
                if p > 0:
                    next_input_id = math_ops.cast(
                                array_ops.reshape(nucleus_sampling(cell_output[:,2:], p), [-1]), dtype=dtype) + 2
                elif k > 0:
                    next_input_id = math_ops.cast(
                                array_ops.reshape(top_k_sampling(cell_output[:,2:], k), [-1]), dtype=dtype) + 2
                else:
                    next_input_id = math_ops.cast(
                            array_ops.reshape(random_ops.multinomial(cell_output[:,2:], 1), [-1]), dtype=dtype) + 2
                done = math_ops.equal(next_input_id, end_of_sequence_id)
                cell_input = array_ops.gather(embeddings, next_input_id)
                context_state = context_state.write(time-1, math_ops.cast(next_input_id, dtype=dtypes.int64))

            # combine cell_input and attention
            next_input = array_ops.concat([cell_input, attention], 1)

            # if time > maxlen, return all true vector
            done = control_flow_ops.cond(
                    math_ops.greater(time, maximum_length),
                    lambda: array_ops.ones([batch_size,], dtype=dtypes.bool),
                    lambda: done)
            return (done, cell_state, next_input, cell_output, context_state)

    return decoder_fn

def attention_decoder_fn_beam_inference(output_fn,
                                       encoder_state,
                                       attention_keys,
                                       attention_values,
                                       attention_score_fn,
                                       attention_construct_fn,
                                       embeddings,
                                       start_of_sequence_id,
                                       end_of_sequence_id,
                                       maximum_length,
                                       num_decoder_symbols,
                                       beam_size,
                                       remove_unk=False,
                                       d_rate=1.0,
                                       dtype=dtypes.int32,
                                       name=None):
    """Attentional decoder function for `dynamic_rnn_decoder` during inference.
    The `attention_decoder_fn_inference` is a simple inference function for a
    sequence-to-sequence model. It should be used when `dynamic_rnn_decoder` is
    in the inference mode.
    The `attention_decoder_fn_inference` is called with user arguments
    and returns the `decoder_fn`, which can be passed to the
    `dynamic_rnn_decoder`, such that
    ```
    dynamic_fn_inference = attention_decoder_fn_inference(...)
    outputs_inference, state_inference = dynamic_rnn_decoder(
            decoder_fn=dynamic_fn_inference, ...)
    ```
    Further usage can be found in the `kernel_tests/seq2seq_test.py`.
    Args:
        output_fn: An output function to project your `cell_output` onto class
        logits.
        An example of an output function;
        ```
            tf.variable_scope("decoder") as varscope
                output_fn = lambda x: layers.linear(x, num_decoder_symbols,
                                                                                        scope=varscope)
                outputs_train, state_train = seq2seq.dynamic_rnn_decoder(...)
                logits_train = output_fn(outputs_train)
                varscope.reuse_variables()
                logits_inference, state_inference = seq2seq.dynamic_rnn_decoder(
                        output_fn=output_fn, ...)
        ```
        If `None` is supplied it will act as an identity function, which
        might be wanted when using the RNNCell `OutputProjectionWrapper`.
        encoder_state: The encoded state to initialize the `dynamic_rnn_decoder`.
        attention_keys: to be compared with target states.
        attention_values: to be used to construct context vectors.
        attention_score_fn: to compute similarity between key and target states.
        attention_construct_fn: to build attention states.
        embeddings: The embeddings matrix used for the decoder sized
        `[num_decoder_symbols, embedding_size]`.
        start_of_sequence_id: The start of sequence ID in the decoder embeddings.
        end_of_sequence_id: The end of sequence ID in the decoder embeddings.
        maximum_length: The maximum allowed of time steps to decode.
        num_decoder_symbols: The number of classes to decode at each time step.
        dtype: (default: `dtypes.int32`) The default data type to use when
        handling integer objects.
        name: (default: `None`) NameScope for the decoder function;
            defaults to "attention_decoder_fn_inference"
    Returns:
        A decoder function with the required interface of `dynamic_rnn_decoder`
        intended for inference.
    """
    with ops.name_scope(name, "attention_decoder_fn_inference", [
            output_fn, encoder_state, attention_keys, attention_values,
            attention_score_fn, attention_construct_fn, embeddings,
            start_of_sequence_id, end_of_sequence_id, maximum_length,
            num_decoder_symbols, dtype
    ]):
        state_size = int(encoder_state[0].get_shape().with_rank(2)[1])
        state = []
        for s in encoder_state:
            state.append(array_ops.reshape(array_ops.concat([array_ops.reshape(s, [-1, 1, state_size])]*beam_size, 1), [-1, state_size]))
        encoder_state = tuple(state)
        origin_batch = array_ops.shape(attention_values)[0]
        attn_length = array_ops.shape(attention_values)[1]
        attention_values = array_ops.reshape(array_ops.concat([array_ops.reshape(attention_values, [-1, 1, attn_length, state_size])]*beam_size, 1), [-1, attn_length, state_size])
        attn_size = array_ops.shape(attention_keys[0])[2] if type(attention_keys) is tuple else array_ops.shape(attention_keys)[2]
        attention_keys = array_ops.reshape(array_ops.concat([array_ops.reshape(attention_keys, [-1, 1, attn_length, attn_size])]*beam_size, 1), [-1, attn_length, attn_size])
        start_of_sequence_id = ops.convert_to_tensor(start_of_sequence_id, dtype)
        end_of_sequence_id = ops.convert_to_tensor(end_of_sequence_id, dtype)
        maximum_length = ops.convert_to_tensor(maximum_length, dtype)
        num_decoder_symbols = ops.convert_to_tensor(num_decoder_symbols, dtype)
        encoder_info = nest.flatten(encoder_state)[0]
        batch_size = encoder_info.get_shape()[0].value
        if output_fn is None:
            output_fn = lambda x: x
        if batch_size is None:
            batch_size = array_ops.shape(encoder_info)[0]
        #beam_size = ops.convert_to_tensor(beam_size, dtype)

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        """Decoder function used in the `dynamic_rnn_decoder` for inference.
        The main difference between this decoder function and the `decoder_fn` in
        `attention_decoder_fn_train` is how `next_cell_input` is calculated. In
        decoder function we calculate the next input by applying an argmax across
        the feature dimension of the output from the decoder. This is a
        greedy-search approach. (Bahdanau et al., 2014) & (Sutskever et al., 2014)
        use beam-search instead.
        Args:
            time: positive integer constant reflecting the current timestep.
            cell_state: state of RNNCell.
            cell_input: input provided by `dynamic_rnn_decoder`.
            cell_output: output of RNNCell.
            context_state: context state provided by `dynamic_rnn_decoder`.
        Returns:
            A tuple (done, next state, next input, emit output, next context state)
            where:
            done: A boolean vector to indicate which sentences has reached a
            `end_of_sequence_id`. This is used for early stopping by the
            `dynamic_rnn_decoder`. When `time>=maximum_length` a boolean vector with
            all elements as `true` is returned.
            next state: `cell_state`, this decoder function does not modify the
            given state.
            next input: The embedding from argmax of the `cell_output` is used as
            `next_input`.
            emit output: If `output_fn is None` the supplied `cell_output` is
            returned, else the `output_fn` is used to update the `cell_output`
            before calculating `next_input` and returning `cell_output`.
            next context state: `context_state`, this decoder function does not
            modify the given context state. The context state could be modified when
            applying e.g. beam search.
        Raises:
            ValueError: if cell_input is not None.
        """
        with ops.name_scope(
                name, "attention_decoder_fn_inference",
                [time, cell_state, cell_input, cell_output, context_state]):
            if cell_input is not None:
                raise ValueError("Expected cell_input to be None, but saw: %s" %
                                                 cell_input)
            if cell_output is None:
                # invariant that this is time == 0
                next_input_id = array_ops.ones(
                        [batch_size,], dtype=dtype) * (start_of_sequence_id)
                done = array_ops.zeros([batch_size,], dtype=dtypes.bool)
                cell_state = encoder_state
                cell_output = array_ops.zeros(
                        [num_decoder_symbols], dtype=dtypes.float32)
                cell_input = array_ops.gather(embeddings, next_input_id)

                # init attention
                attention = _init_attention(encoder_state)
                # init context state
                log_beam_probs = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name="log_beam_probs", size=maximum_length, dynamic_size=True, infer_shape=False)
                beam_parents = tensor_array_ops.TensorArray(dtype=dtypes.int32, tensor_array_name="beam_parents", size=maximum_length, dynamic_size=True, infer_shape=False)
                beam_symbols = tensor_array_ops.TensorArray(dtype=dtypes.int32, tensor_array_name="beam_symbols", size=maximum_length, dynamic_size=True, infer_shape=False)
                result_probs = tensor_array_ops.TensorArray(dtype=dtypes.float32, tensor_array_name="result_probs", size=maximum_length, dynamic_size=True, infer_shape=False)
                result_parents = tensor_array_ops.TensorArray(dtype=dtypes.int32, tensor_array_name="result_parents", size=maximum_length, dynamic_size=True, infer_shape=False)
                result_symbols = tensor_array_ops.TensorArray(dtype=dtypes.int32, tensor_array_name="result_symbols", size=maximum_length, dynamic_size=True, infer_shape=False)
                context_state = (log_beam_probs, beam_parents, beam_symbols, result_probs, result_parents, result_symbols)
            else:
                # construct attention
                attention = attention_construct_fn(cell_output, attention_keys,
                        attention_values)
                cell_output = attention

                # beam search decoder
                (log_beam_probs, beam_parents, beam_symbols, result_probs, result_parents, result_symbols) = context_state
                
                cell_output = output_fn(cell_output)    # logits
                cell_output = nn_ops.softmax(cell_output)
                

                cell_output = array_ops.split(cell_output, [2, num_decoder_symbols-2], 1)[1]
                #k_indices = tf.argsort(cell_output, direction='DESCENDING')
                _, k_indices = nn_ops.top_k(cell_output, num_decoder_symbols-2)
                updates = array_ops.reshape(array_ops.tile(array_ops.expand_dims(math_ops.cast(math_ops.range(num_decoder_symbols-2)+1, dtypes.float32), axis=0), [batch_size, 1]), [-1]) * d_rate
                k_indices_0 = array_ops.reshape(array_ops.tile(array_ops.expand_dims(math_ops.range(batch_size), axis=-1), [1, num_decoder_symbols-2]), [-1, 1])
                k_indices_1 = array_ops.reshape(k_indices, [-1, 1])
                k_indices = array_ops.concat([k_indices_0, k_indices_1], axis=1)
                penalty = array_ops.scatter_nd(k_indices, updates, [batch_size, num_decoder_symbols-2])

                tmp_output = array_ops.gather(cell_output, math_ops.range(origin_batch)*beam_size)

                probs = control_flow_ops.cond(
                        math_ops.equal(time, ops.convert_to_tensor(1, dtype)),
                        lambda: math_ops.log(tmp_output+ops.convert_to_tensor(1e-20, dtypes.float32)),
                        lambda: math_ops.log(cell_output+ops.convert_to_tensor(1e-20, dtypes.float32)) + array_ops.reshape(log_beam_probs.read(time-2), [-1, 1]) - penalty)

                probs = array_ops.reshape(probs, [origin_batch, -1])
                best_probs, indices = nn_ops.top_k(probs, beam_size * 2)
                #indices = array_ops.reshape(indices, [-1])
                indices_flatten = array_ops.reshape(indices, [-1]) + array_ops.reshape(array_ops.concat([array_ops.reshape(math_ops.range(origin_batch)*((num_decoder_symbols-2)*beam_size), [-1, 1])]*(beam_size*2), 1), [origin_batch*beam_size*2])
                best_probs_flatten = array_ops.reshape(best_probs, [-1])

                symbols = indices_flatten % (num_decoder_symbols - 2)
                symbols = symbols + 2
                parents = indices_flatten // (num_decoder_symbols - 2)

                probs_wo_eos = best_probs + 1e5*math_ops.cast(math_ops.cast((indices%(num_decoder_symbols-2)+2)-end_of_sequence_id, dtypes.bool), dtypes.float32)
                
                best_probs_wo_eos, indices_wo_eos = nn_ops.top_k(probs_wo_eos, beam_size)

                indices_wo_eos = array_ops.reshape(indices_wo_eos, [-1]) + array_ops.reshape(array_ops.concat([array_ops.reshape(math_ops.range(origin_batch)*(beam_size*2), [-1, 1])]*beam_size, 1), [origin_batch*beam_size])

                _probs = array_ops.gather(best_probs_flatten, indices_wo_eos)
                _symbols = array_ops.gather(symbols, indices_wo_eos)
                _parents = array_ops.gather(parents, indices_wo_eos)


                log_beam_probs = log_beam_probs.write(time-1, _probs)
                beam_symbols = beam_symbols.write(time-1, _symbols)
                beam_parents = beam_parents.write(time-1, _parents)
                result_probs = result_probs.write(time-1, best_probs_flatten)
                result_symbols = result_symbols.write(time-1, symbols)
                result_parents = result_parents.write(time-1, parents)


                next_input_id = array_ops.reshape(_symbols, [batch_size])

                state_size = int(cell_state[0].get_shape().with_rank(2)[1])
                attn_size = int(attention.get_shape().with_rank(2)[1])
                state = []
                for j in cell_state:
                    state.append(array_ops.reshape(array_ops.gather(j, _parents), [-1, state_size]))
                cell_state = tuple(state)
                attention = array_ops.reshape(array_ops.gather(attention, _parents), [-1, attn_size])

                done = math_ops.equal(next_input_id, end_of_sequence_id)
                cell_input = array_ops.gather(embeddings, next_input_id)

            # combine cell_input and attention
            next_input = array_ops.concat([cell_input, attention], 1)

            # if time > maxlen, return all true vector
            done = control_flow_ops.cond(
                    math_ops.greater(time, maximum_length),
                    lambda: array_ops.ones([batch_size,], dtype=dtypes.bool),
                    lambda: array_ops.zeros([batch_size,], dtype=dtypes.bool))
            return (done, cell_state, next_input, cell_output, (log_beam_probs, beam_parents, beam_symbols, result_probs, result_parents, result_symbols))#context_state)

    return decoder_fn

## Helper functions ##
def prepare_attention(attention_states,
                          attention_option,
                          num_units,
                          mem=None,
                          mask=None,
                          output_alignments=False,
                          reuse=False):
    """Prepare keys/values/functions for attention.
    Args:
        attention_states: hidden states to attend over.
        attention_option: how to compute attention, either "luong" or "bahdanau".
        num_units: hidden state dimension.
        reuse: whether to reuse variable scope.
    Returns:
        attention_keys: to be compared with target states.
        attention_values: to be used to construct context vectors.
        attention_score_fn: to compute similarity between key and target states.
        attention_construct_fn: to build attention states.
    """

    # Prepare attention keys / values from attention_states
    with variable_scope.variable_scope("attention_keys", reuse=reuse) as scope:
        attention_keys = layers.linear(
            attention_states, num_units, biases_initializer=None, scope=scope)
    attention_values = attention_states

    if mem is not None:
        if type(mem) is tuple:
            with variable_scope.variable_scope("mem_keys", reuse=reuse) as scope:
                attention_keys2 = layers.linear(array_ops.concat([mem[0], mem[1]], axis=-1),
                    num_units, biases_initializer=None, scope=scope)
            with variable_scope.variable_scope("mem_values", reuse=reuse) as scope:
                attention_states2 = layers.linear(mem[2], num_units, 
                        biases_initializer=None, scope=scope)
            attention_keys = (attention_keys, attention_keys2, mask)
            attention_values = (attention_states, attention_states2)

        

    # Attention score function
    if mem is None:
        attention_score_fn = _create_attention_score_fn("attention_score", num_units,
                                                            attention_option, reuse)
    else:
        attention_score_fn = (_create_attention_score_fn("attention_score", num_units,
                                                            attention_option, reuse),
                            _create_attention_score_fn("mem_score", num_units,
                                                            "luong", reuse, output_alignments=output_alignments))

    # Attention construction function
    attention_construct_fn = _create_attention_construct_fn("attention_construct",
                                  num_units,
                                  attention_score_fn,
                                  reuse)

    return (attention_keys, attention_values, attention_score_fn,
                    attention_construct_fn)


def _init_attention(encoder_state):
    """Initialize attention. Handling both LSTM and GRU.
    Args:
        encoder_state: The encoded state to initialize the `dynamic_rnn_decoder`.
    Returns:
        attn: initial zero attention vector.
    """

    # Multi- vs single-layer
    # TODO(thangluong): is this the best way to check?
    if isinstance(encoder_state, tuple):
        top_state = encoder_state[-1]
    else:
        top_state = encoder_state
    
    '''
    # LSTM vs GRU
    if isinstance(top_state, tf.nn.rnn_cell.LSTMStateTuple):
        attn = array_ops.zeros_like(top_state.h)
    else:
        attn = array_ops.zeros_like(top_state)
    '''
    attn = array_ops.zeros_like(top_state)

    return attn


def _create_attention_construct_fn(name, num_units, attention_score_fn, reuse):
    """Function to compute attention vectors.
    Args:
        name: to label variables.
        num_units: hidden state dimension.
        attention_score_fn: to compute similarity between key and target states.
        reuse: whether to reuse variable scope.
    Returns:
        attention_construct_fn: to build attention states.
    """
    with variable_scope.variable_scope(name, reuse=reuse) as scope:

        def construct_fn(attention_query, attention_keys, attention_values):
            alignments = None
            if type(attention_score_fn) is tuple:
                context0 = attention_score_fn[0](attention_query, attention_keys[0],
                                                                         attention_values[0])
                #concat_input = array_ops.concat([attention_query, context0, context1], 1)
                concat_input = array_ops.concat([attention_query, context0], 1)
                attention = layers.linear(
                        concat_input, num_units, biases_initializer=None, scope=scope)
                #context1 = attention_score_fn[1](attention_query, attention_keys[1],
                context1 = attention_score_fn[1](attention, attention_keys[1],
                                                                         attention_values[1], mask=attention_keys[2])
                context1, alignments = context1
            else:
                context = attention_score_fn(attention_query, attention_keys,
                                                                         attention_values)
                concat_input = array_ops.concat([attention_query, context], 1)
                attention = layers.linear(
                        concat_input, num_units, biases_initializer=None, scope=scope)
            if alignments is None:
                return attention
            else:
                return attention, alignments

        return construct_fn


# keys: [batch_size, attention_length, attn_size]
# query: [batch_size, 1, attn_size]
# return weights [batch_size, attention_length]
@function.Defun(func_name="attn_add_fun", noinline=True)
def _attn_add_fun(v, keys, query):
    return math_ops.reduce_sum(v * math_ops.tanh(keys + query), [2])


@function.Defun(func_name="attn_mul_fun", noinline=True)
def _attn_mul_fun(keys, query):
    return math_ops.reduce_sum(keys * query, [2])


def _create_attention_score_fn(name,
                                   num_units,
                                   attention_option,
                                   reuse,
                                   output_alignments=False,
                                   dtype=dtypes.float32):
    """Different ways to compute attention scores.
    Args:
        name: to label variables.
        num_units: hidden state dimension.
        attention_option: how to compute attention, either "luong" or "bahdanau".
            "bahdanau": additive (Bahdanau et al., ICLR'2015)
            "luong": multiplicative (Luong et al., EMNLP'2015)
        reuse: whether to reuse variable scope.
        dtype: (default: `dtypes.float32`) data type to use.
    Returns:
        attention_score_fn: to compute similarity between key and target states.
    """
    with variable_scope.variable_scope(name, reuse=reuse):
        if attention_option == "bahdanau":
            query_w = variable_scope.get_variable(
                    "attnW", [num_units, num_units], dtype=dtype)
            score_v = variable_scope.get_variable("attnV", [num_units], dtype=dtype)


        def attention_score_fn(query, keys, values, mask=None):
            """Put attention masks on attention_values using attention_keys and query.
            Args:
                query: A Tensor of shape [batch_size, num_units].
                keys: A Tensor of shape [batch_size, attention_length, num_units].
                values: A Tensor of shape [batch_size, attention_length, num_units].
            Returns:
                context_vector: A Tensor of shape [batch_size, num_units].
            Raises:
                ValueError: if attention_option is neither "luong" or "bahdanau".
            """
            if attention_option == "bahdanau":
                # transform query
                query = math_ops.matmul(query, query_w)

                # reshape query: [batch_size, 1, num_units]
                query = array_ops.reshape(query, [-1, 1, num_units])

                # attn_fun
                scores = _attn_add_fun(score_v, keys, query)
            elif attention_option == "luong":
                # reshape query: [batch_size, 1, num_units]
                query = array_ops.reshape(query, [-1, 1, num_units])

                # attn_fun
                scores = _attn_mul_fun(keys, query)
            else:
                raise ValueError("Unknown attention option %s!" % attention_option)

            if mask is None:
                alignments = nn_ops.softmax(scores)
            else:
                alignments = nn_ops.softmax(scores + mask)
            #alignments = tf.Print(alignments, [alignments], summarize=1000)

            # Now calculate the attention-weighted vector.
            alignments_expand = array_ops.expand_dims(alignments, 2)
            context_vector = math_ops.reduce_sum(alignments_expand * values, [1])
            context_vector.set_shape([None, num_units])

            if output_alignments:
                return context_vector, alignments
            return context_vector

        return attention_score_fn
