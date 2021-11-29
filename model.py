import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.seq2seq.python.ops.loss import sequence_loss
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.contrib.layers.python.layers import layers
from dynamic_decoder import dynamic_rnn_decoder
from output_projection import output_projection_layer
from attention_decoder import * 
from tensorflow.contrib.session_bundle import exporter

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

class Seq2SeqModel(object):
    def __init__(self,
            num_symbols,
            num_embed_units,
            num_units,
            num_layers,
            beam_size,
            embed,
            embed_kg=None,
            learning_rate=0.5,
            remove_unk=False,
            learning_rate_decay_factor=0.95,
            max_gradient_norm=5.0,
            num_samples=512,
            max_length=20,
            beam_max_length=20,
            output_alignments=True,
            mask_use=True,
            use_lstm=False):
        
        self.posts = tf.placeholder(tf.string, (None, None), 'enc_inps')  # batch*len
        self.posts_length = tf.placeholder(tf.int32, (None), 'enc_lens')  # batch
        self.responses = tf.placeholder(tf.string, (None, None), 'dec_inps')  # batch*len
        self.responses_length = tf.placeholder(tf.int32, (None), 'dec_lens')  # batch
        self.kgs_h = tf.placeholder(tf.string, (None, None, None), 'kg_h_inps')  # batch*triple_num*len
        self.kgs_r = tf.placeholder(tf.string, (None, None, None), 'kg_r_inps')  # batch*triple_num*len
        self.kgs_head_r = tf.placeholder(tf.string, (None, None, None), 'kg_head_r_inps')  # batch*triple_num*len
        self.kgs_head_r_mask = tf.placeholder(tf.float32, (None, None, None), 'kg_head_r_mask')  # batch*triple_num
        self.kgs_h_index = tf.placeholder(tf.int32, (None, None, 2), 'kg_h_idxs')  # batch*triple_num
        self.kgs_h_index_ga = tf.placeholder(tf.int32, (None, None, 2), 'kg_h_idxs_ga')  # batch*triple_num
        self.kgs_r_length = tf.placeholder(tf.int32, (None, None), 'kg_r_lens')  # batch*triple_num
        self.kgs_t = tf.placeholder(tf.string, (None, None, None), 'kg_t_inps')  # batch*triple_num*len
        self.match_triples = tf.placeholder(tf.int32, (None, None), 'match_triples')  # batch*triple_num
        encoder_batch_size = tf.shape(self.posts)[0]
        triple_num = tf.shape(self.kgs_h)[1]
        one_hot_triples = tf.one_hot(self.match_triples, triple_num + tf.shape(self.kgs_head_r)[1])
        use_triples = tf.reduce_sum(one_hot_triples, axis=2)
        
        self.symbol2index = MutableHashTable(
                key_dtype=tf.string,
                value_dtype=tf.int64,
                default_value=UNK_ID,
                name="in_table",
                checkpoint=True)
        self.index2symbol = MutableHashTable(
                key_dtype=tf.int64,
                value_dtype=tf.string,
                default_value='_UNK',
                name="out_table",
                checkpoint=True)
        # build the vocab table (string to index)
        # initialize the training process
        self.learning_rate = tf.Variable(float(learning_rate), 
                trainable=False, dtype=tf.float32)
        self.ppx_best = tf.Variable(float('inf'), 
                trainable=False, dtype=tf.float32, name='ppx_best')
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)


        self.posts_input = self.symbol2index.lookup(self.posts)   # batch*len
        self.responses_target = self.symbol2index.lookup(self.responses)   #batch*len
        
        batch_size, decoder_len = tf.shape(self.responses)[0], tf.shape(self.responses)[1]
        self.responses_input = tf.concat([tf.ones([batch_size, 1], dtype=tf.int64)*GO_ID,
            tf.split(self.responses_target, [decoder_len-1, 1], 1)[0]], 1)   # batch*len
        self.decoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.responses_length-1, 
            decoder_len), reverse=True, axis=1), [-1, decoder_len])
        
        # build the embedding table (index to vector)
        if embed is None:
            # initialize the embedding randomly
            self.embed = tf.get_variable('embed', [num_symbols, num_embed_units], tf.float32)
        else:
            # initialize the embedding by pre-trained word vectors
            self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)

        encoder_fw_cell = MultiRNNCell([GRUCell(num_units // 2) for _ in range(num_layers)])
        encoder_bw_cell = MultiRNNCell([GRUCell(num_units // 2) for _ in range(num_layers)])
        decoder_cell = MultiRNNCell([GRUCell(num_units) for _ in range(num_layers)])
        path_cell = GRUCell(num_embed_units)

        def get_bidirectional_res(res):
            outputs, states = res
            return tf.concat(outputs, axis=2), tuple([tf.concat([state_fw, state_bw], axis=1) for state_fw, state_bw in zip(states[0], states[1])])
        
        self.encoder_input = tf.nn.embedding_lookup(self.embed, self.posts_input) #batch*len*unit
        # rnn encoder
        with tf.variable_scope('encoder'):
            encoder_output, encoder_state = get_bidirectional_res(tf.nn.bidirectional_dynamic_rnn(encoder_fw_cell, encoder_bw_cell, self.encoder_input, 
                    self.posts_length, dtype=tf.float32))

        with tf.variable_scope('graph_attention'):
            kgs_r_vector = tf.reshape(tf.nn.embedding_lookup(self.embed, self.symbol2index.lookup(self.kgs_head_r)), [encoder_batch_size, tf.shape(self.kgs_head_r)[1], tf.shape(self.kgs_head_r)[2], num_embed_units])
            head = tf.layers.dense(encoder_state[-1], num_embed_units, activation=tf.tanh)
            head_emb = tf.tile(tf.reshape(head, [encoder_batch_size, 1, num_embed_units]), [1, tf.shape(self.kgs_head_r)[1], 1])
            head = tf.tile(tf.reshape(head, [encoder_batch_size, 1, 1, num_embed_units]), [1, tf.shape(self.kgs_head_r)[1], tf.shape(self.kgs_head_r)[2], 1])
            e_weight = tf.reduce_sum(head * kgs_r_vector, axis=-1) - 1e10 * (1.0 - self.kgs_head_r_mask)
            alpha_weight = tf.expand_dims(tf.nn.softmax(e_weight), axis=-1)
            head_emb = tf.reduce_sum(alpha_weight * tf.concat([head, kgs_r_vector], axis=-1), axis=2)
            head_emb = tf.reshape(tf.layers.dense(head_emb, num_embed_units, activation=tf.tanh), [encoder_batch_size, tf.shape(self.kgs_head_r)[1], num_embed_units])
            tail_emb = tf.layers.dense(head_emb, num_embed_units, activation=tf.tanh)
            self.kgs_h_vector_ga = tf.gather_nd(tail_emb, self.kgs_h_index_ga)


        self.kgs_h_vector = tf.layers.dense(tf.gather_nd(encoder_output, self.kgs_h_index), num_embed_units, activation=tf.tanh)


        self.kgs_r_vector = tf.reshape(tf.nn.embedding_lookup(self.embed, self.symbol2index.lookup(self.kgs_r)), [encoder_batch_size, triple_num, num_embed_units])

        state = tf.reshape(self.kgs_h_vector, [encoder_batch_size * triple_num, num_embed_units])
        self.kgs_t_vector, state = path_cell(tf.reshape(self.kgs_r_vector, [encoder_batch_size * triple_num, num_embed_units]), state)
        self.kgs_t_vector = tf.layers.dense(tf.reshape(self.kgs_t_vector, [encoder_batch_size, triple_num, num_embed_units]), num_embed_units, activation=tf.tanh)
        self.kgs_mask = -1e10 * tf.cast(tf.equal(self.kgs_r_length, 0), tf.float32)

        self.kgs_h_vector = tf.concat([self.kgs_h_vector, head_emb], axis=1)
        self.kgs_t_vector = tf.concat([self.kgs_t_vector, tail_emb], axis=1)
        self.kgs_mask = tf.concat([self.kgs_mask, -1e10 * (1.0 - tf.reduce_max(self.kgs_head_r_mask, axis=-1))], axis=1)

        self.kgs_vector = (self.kgs_h_vector, self.kgs_t_vector, self.kgs_t_vector)

        

        use_triples_expand = tf.expand_dims(use_triples, axis=-1)
        decoder_input = (1.0 - use_triples_expand) * tf.nn.embedding_lookup(self.embed, self.responses_target) 
        entity_input = use_triples_expand * tf.einsum('blt,bth->blh', one_hot_triples, self.kgs_t_vector)
        start_input = tf.nn.embedding_lookup(self.embed, tf.ones([batch_size, 1], dtype=tf.int64)*GO_ID)
        self.decoder_input = tf.concat([start_input, entity_input + decoder_input], axis=1)[:,:-1,:]




        # get output projection function
        output_fn, selector_fn, sequence_loss, sampled_sequence_loss, total_loss = output_projection_layer(num_units, 
                num_symbols, num_samples)

        

        with tf.variable_scope('decoder'):
            # get attention function
            attention_keys_init, attention_values_init, attention_score_fn_init, attention_construct_fn_init \
                    = prepare_attention(encoder_output, 'bahdanau', num_units, mem=self.kgs_vector, mask=self.kgs_mask, output_alignments=output_alignments)

            decoder_fn_train = attention_decoder_fn_train(
                    encoder_state, attention_keys_init, attention_values_init,
                    attention_score_fn_init, attention_construct_fn_init, triple_num, output_alignments=output_alignments, max_length=tf.reduce_max(self.responses_length))
            self.decoder_output, _, alignments_ta = dynamic_rnn_decoder(decoder_cell, decoder_fn_train, 
                    self.decoder_input, self.responses_length, scope="decoder_rnn")
            if output_alignments: 
                self.alignments = tf.transpose(alignments_ta.stack(), perm=[1,0,2])
                self.decoder_loss, self.ppx_loss, self.sentence_ppx = total_loss(self.decoder_output, self.responses_target, self.decoder_mask, self.alignments, self.kgs_t_vector, use_triples, one_hot_triples)
                self.sentence_ppx = tf.identity(self.sentence_ppx, name='ppx_loss')
            else:
                self.decoder_loss = sequence_loss(self.decoder_output, 
                        self.responses_target, self.decoder_mask)
        
        with tf.variable_scope('decoder', reuse=True):
            # get attention function
            attention_keys, attention_values, attention_score_fn, attention_construct_fn \
                    = prepare_attention(encoder_output, 'bahdanau', num_units, reuse=True, mem=self.kgs_vector, mask=self.kgs_mask, output_alignments=output_alignments)
            decoder_fn_inference = attention_decoder_fn_inference(
                    output_fn, encoder_state, attention_keys, attention_values, 
                    attention_score_fn, attention_construct_fn, self.embed, GO_ID, 
                    EOS_ID, max_length, num_symbols, imem=self.kgs_t_vector, selector_fn=selector_fn)
                
            if output_alignments:
                self.decoder_distribution, _, (_, output_ids_ta) = dynamic_rnn_decoder(decoder_cell,
                        decoder_fn_inference, scope="decoder_rnn")

                output_len = tf.shape(self.decoder_distribution)[1]
                output_ids = tf.transpose(output_ids_ta.gather(tf.range(output_len)))
                word_ids = tf.cast(tf.clip_by_value(output_ids, 0, num_symbols), tf.int64)
                entity_ids = tf.reshape(tf.clip_by_value(-output_ids, 0, num_symbols) + tf.reshape(tf.range(encoder_batch_size) * tf.shape(self.kgs_t_vector)[1], [-1, 1]), [-1])
                entities = tf.reshape(tf.gather(tf.reshape(self.kgs_t, [-1]), entity_ids), [-1, output_len])
                words = self.index2symbol.lookup(word_ids)
                self.generation = tf.where(output_ids > 0, words, entities)
                self.generation = tf.identity(self.generation, name='generation')
            else:
                self.decoder_distribution, _, _ = dynamic_rnn_decoder(decoder_cell,
                        decoder_fn_inference, scope="decoder_rnn")
                self.generation_index = tf.argmax(tf.split(self.decoder_distribution,
                    [2, num_symbols-2], 2)[1], 2) + 2 # for removing UNK
                self.generation = self.index2symbol.lookup(self.generation_index, name='generation') 

                decoder_fn_sample = attention_decoder_fn_sample(
                        output_fn, encoder_state, attention_keys, attention_values, 
                        attention_score_fn, attention_construct_fn, self.embed, GO_ID, 
                        EOS_ID, max_length, num_symbols)
                    
                self.sample_distribution, _, context_state = dynamic_rnn_decoder(decoder_cell,
                        decoder_fn_sample, scope='decoder_rnn')
                self.sample = self.index2symbol.lookup(tf.transpose(context_state.gather(tf.range(tf.shape(self.sample_distribution)[1]))), name='sample')


        self.params = tf.trainable_variables()
            
        # calculate the gradient of parameters
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients = tf.gradients(self.decoder_loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 
                max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), 
                global_step=self.global_step)
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, 
                max_to_keep=20, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

        self.saver_best = tf.train.Saver(write_version=tf.train.SaverDef.V2, 
                max_to_keep=10, pad_step_number=True)
        

    def print_parameters(self):
        num_param = 0
        for item in self.params:
            n = 1
            for dim in item.get_shape():
                n *= dim
            num_param += n
            print('%s: %s' % (item.name, item.get_shape()))
        print('total param: %d' % num_param)
    
    def step_decoder(self, session, data, forward_only=False, inference=False):
        input_feed = {self.posts: data['posts'],
                self.posts_length: data['posts_length'],
                self.responses: data['responses'],
                self.responses_length: data['responses_length'],
                self.kgs_h: data['kgs_h'],
                self.kgs_r: data['kgs_r'],
                self.kgs_t: data['kgs_t'],
                self.kgs_h_index: data['kgs_h_index'],
                self.kgs_r_length: data['kgs_r_length'],
                self.match_triples: data['match_triples'],
                self.kgs_h_index_ga: data['kgs_h_index_ga'],
                self.kgs_head_r: data['kgs_head_r'],
                self.kgs_head_r_mask: data['kgs_head_r_mask'],
                }
        if inference:
            output_feed = [self.generation]
        elif forward_only:
            output_feed = [self.sentence_ppx]
        else:
            output_feed = [self.decoder_loss, self.gradient_norm, self.update]
        return session.run(output_feed, input_feed)
