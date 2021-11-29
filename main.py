#coding=utf8
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from collections import Counter
import nltk
import sys
import io
import time
import random
import jieba
import json
random.seed(1229)

from model import Seq2SeqModel, _START_VOCAB
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
np.set_printoptions(threshold=np.inf)

tf.app.flags.DEFINE_boolean("is_train", False, "Set to False to inference.")
tf.app.flags.DEFINE_integer("symbols", 30000, "vocabulary size.")
tf.app.flags.DEFINE_integer("set_size", 900, "Size of each model layer.")
tf.app.flags.DEFINE_integer("max_length", 60, "Max length of response.")
tf.app.flags.DEFINE_integer("embed_units", 300, "Size of word embedding.")
tf.app.flags.DEFINE_integer("units", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("triple_num", 20, "Num of triples.")
tf.app.flags.DEFINE_integer("layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("beam_size", 20, "Beam size to use during beam inference.")
tf.app.flags.DEFINE_boolean("beam_use", False, "use beam search or not.")
tf.app.flags.DEFINE_boolean("mask_use", True, "use masked kg or not.")
tf.app.flags.DEFINE_boolean("mem_use", False, "use memory or not.")
tf.app.flags.DEFINE_boolean("post_process", False, "use post process or not.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "./data/duconv", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_integer("per_checkpoint", 1000, "How many steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("inference_version", 0, "The version for inferencing.")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")
tf.app.flags.DEFINE_string("inference_path", "", "Set filename of inference, default isscreen")

FLAGS = tf.app.flags.FLAGS
kb = {}

def load_data(path, fname):
    global kb
    raw = []
    kg_all = []
    with open('%s/%s' % (path, fname), encoding='utf8') as f:
        kb = json.loads(f.readline().strip())
        for idx, line in enumerate(f):
            if idx == 100:
                pass#break
            mtknz_post = nltk.tokenize.MWETokenizer(separator='_')
            mtknz_resp = nltk.tokenize.MWETokenizer(separator='_')

            content = json.loads(line.strip())
            context = []
            session = []
            kg = []
            entity2idx = {}
            ents = {}
            for ent in content['kg']:
                mtknz_post.add_mwe(nltk.word_tokenize(ent.lower()))
                goldens = content['kg'][ent]
                kg += [goldens[:]]
                for triple in kb[ent]:
                    if triple not in goldens:
                        kg[-1] += [triple[:]]
                        if len(kg[-1]) == FLAGS.triple_num:
                            break
                for (h, r, t) in kg[-1]:
                    mtknz_resp.add_mwe(nltk.word_tokenize(h.lower()))
                    mtknz_resp.add_mwe(nltk.word_tokenize(t.lower()))

            eidx = 0
            for triples in kg:
                ent = mtknz_post.tokenize(nltk.word_tokenize(triples[0][0].lower()))
                if ent[0] not in ents:
                    ents[ent[0]] = len(ents)
                entity2idx[ent[0]] = -1
                for triple in triples:
                    triple[0] = mtknz_resp.tokenize(nltk.word_tokenize(triple[0].lower()))
                    triple[2] = mtknz_resp.tokenize(nltk.word_tokenize(triple[2].lower()))
                    if len(triple[2]) == 0:
                        triple[2] = ['']
                        continue
                    entity2idx[triple[2][0]] = eidx
                    eidx += 1

            context += [mtknz_resp.tokenize(nltk.word_tokenize(content['dialog'][0][0].lower()))]
            for turn in content['dialog'][1:]:
                resp = mtknz_resp.tokenize(nltk.word_tokenize(turn[0].lower()))[:FLAGS.max_length]
                post = [x for t in context for x in t]
                #if len(resp) > 0 and (not any([ent in resp for ent in entity2idx]) or any([ent in post for ent in ents])):
                if len(resp) > 0:
                    session += [{'post': post, 'response': resp, 'kg': kg, 'entity2idx': entity2idx, 'head_ents': ents}]
                context += [['_EOS'] + mtknz_resp.tokenize(nltk.word_tokenize(turn[0].lower()))]
            if session:
                raw += [session]
    
    data_train, data_valid, data_test_seen, data_test_unseen = raw[:-4 * FLAGS.set_size], raw[-4 * FLAGS.set_size:-2 * FLAGS.set_size], raw[-2 * FLAGS.set_size: -FLAGS.set_size], raw[-FLAGS.set_size:]
    return data_train, data_valid, data_test_seen, data_test_unseen

def build_vocab(path, data):
    print("Creating vocabulary...")
    vocab = {}
    vocab_kg = {}
    for i, session in enumerate(data):
        if i % 1000 == 0:
            print("    processing line %d" % i)
        pair = session[-1]
        for triples in pair['kg']:
            for triple in triples:
                for token in triple[0] + [triple[1]] + triple[2]:
                    if token in vocab_kg:
                        vocab_kg[token] += 1
                    else:
                        vocab_kg[token] = 1
                vocab[triple[1]] = vocab.get(triple[1], 0) + 1
        for token in session[-1]['post'] + [x for turn in session for x in turn['response']]:
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1
    vocab_kg_list = sorted(vocab_kg, key=vocab_kg.get, reverse=True)
    vocab_list = _START_VOCAB + ['_MASK%d' % x for x in range(16)] + [w for w in sorted(vocab, key=vocab.get, reverse=True) if w[-4:] == '/rel' or '_' not in w] + [ent for ent in vocab_kg_list if ent not in vocab]

    if len(vocab_list) > FLAGS.symbols:
        vocab_list = vocab_list[:FLAGS.symbols]
    else:
        FLAGS.symbols = len(vocab_list)

    print("Loading word vectors...")
    vectors = {}
    with open(path, encoding='utf8') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("    processing line %d" % i)
            s = line.strip()
            word = s[:s.find(' ')]
            vector = s[s.find(' ')+1:]
            vectors[word] = vector
    
    embed = []
    for word in vocab_list:
        if word in vectors:
            vector = list(map(float, vectors[word].split()))
        else:
            vector = np.zeros((FLAGS.embed_units), dtype=np.float32)
        embed.append(vector)
    embed = np.array(embed, dtype=np.float32)
            
    return vocab_list, embed

def gen_batched_data(data):
    def padding(sent, l, eos=True):
        return sent + (['_EOS'] if eos else []) + ['_PAD'] * (l-len(sent)-1)

    exist_in_post = lambda x, y: -1 if x == -1 or y[x] == 0 else x
    
    encoder_len, decoder_len, triple_num, triple_num_s, head_len, tail_len, head_num = 0, 0, 0, 0, 0, 0, 0
    posts, responses, posts_length, responses_length = [], [], [], []
    kgs, kgs_length = {'h': [], 'r': [], 't': [], 'head_r': [], 'head_r_mask': [], 'h_index_ga': []}, {'h': [], 'r': [], 't': []}
    match_triples = []

    for item in data:
        encoder_len = max(encoder_len, len(item['post']))
        decoder_len = max(decoder_len, len(item['response']))
        _triple_num = 0
        kgs_length['h'] += [[]]
        kgs_length['r'] += [[]]
        kgs_length['t'] += [[]]
        for tris in item['kg']:
            _triple_num += len(tris)
            triple_num_s = max(triple_num_s, len(tris))
            for h, r, t in tris:
                kgs_length['h'][-1] += [len(h)]
                kgs_length['r'][-1] += [1]
                kgs_length['t'][-1] += [len(t)]
                head_len = max(head_len, len(h))
                tail_len = max(tail_len, len(t))
        head_num = max(head_num, len(item['kg']))
        triple_num = max(triple_num, _triple_num)
    encoder_len += 1
    decoder_len += 1
    head_len += 1
    tail_len += 1

    for idx, item in enumerate(data):
        midx = 0
        entity2mask = {}
        entity2idx = {}
        for i, word in enumerate(item['post'][::-1]):
            if word in item['head_ents'] or word in item['entity2idx']:
                if word not in entity2mask:
                    entity2mask[word] = '_MASK%d' % midx
                    midx += 1
                if word not in entity2idx:
                    entity2idx[word] = len(item['post']) - 1 - i

        posts.append(padding([entity2mask.get(word, word) for word in item['post']], encoder_len))
        responses.append(padding(item['response'], decoder_len))
        posts_length.append(len(item['post'])+1)
        responses_length.append(len(item['response'])+1)
        kgs['h'] += [[]]
        kgs['r'] += [[]]
        kgs['t'] += [[]]
        kgs['head_r'] += [[]]
        kgs['head_r_mask'] += [[]]
        kgs['h_index_ga'] += [[]]
        kgs_length['h'][idx] = []
        head_ents = []
        for tdx, triples in enumerate(item['kg']):
            kgs['head_r'][-1] += [[]]
            kgs['head_r_mask'][-1] += [[]]
            for h, r, t in triples:
                if h[0] in entity2idx:
                    kgs_length['h'][idx] += [[idx, entity2idx[h[0]]]]
                else:
                    kgs_length['h'][idx] += [[idx, 0]]
                    kgs_length['r'][idx][len(kgs_length['h'][idx])-1] = 0
                kgs['h'][-1] += [padding([entity2mask.get(word, '_UNK') if FLAGS.mask_use else word for word in h], head_len, False)]
                kgs['r'][-1] += [[r]]
                kgs['t'][-1] += [padding(t, tail_len, False)]
                kgs['h_index_ga'][-1] += [[idx, tdx]]
                kgs['head_r'][-1][-1] += [r]
                kgs['head_r_mask'][-1][-1] += [1.]
                if h[0] not in head_ents:
                    head_ents += [h[0]]
            for i in range(triple_num_s - len(triples)):
                kgs['head_r'][-1][-1] += ['']
                kgs['head_r_mask'][-1][-1] += [0.]
        for i in range(head_num - len(kgs['head_r'][-1])):
            kgs['head_r'][-1] += [[]]
            kgs['head_r_mask'][-1] += [[]]
            for j in range(triple_num_s):
                kgs['head_r'][-1][-1] += ['']
                kgs['head_r_mask'][-1][-1] += [0.]
                
        for h, r, t in [[[], '', []] for _ in range(triple_num - len(kgs['h'][-1]))]:
            kgs['h'][-1] += [padding(h, head_len, False)]
            kgs['r'][-1] += [[r]]
            kgs['t'][-1] += [padding(t, tail_len, False)]
            kgs['h_index_ga'][-1] += [[idx, 0]]

        for i, ent in enumerate(head_ents):
            item['head_ents'][ent] = triple_num + i

        for ent, i in list(sorted(item['head_ents'].items(), key=lambda x: x[1])) + [['', 0] for _ in range(head_num-len(item['head_ents']))]:
            kgs['t'][-1] += [padding([ent], tail_len, False)]

        for k in 'hrt':
            for kg_length in kgs_length[k]:
                kg_length += [[0, 0] if k == 'h' else 0 for _ in range(triple_num - len(kg_length))]
        match_triples.append([item['head_ents'][word] if word in item['head_ents'] else exist_in_post(item['entity2idx'].get(word, -1), kgs_length['r'][idx]) for word in responses[-1]])



    batched_data = {'posts': np.array(posts),
            'responses': np.array(responses),
            'posts_length': posts_length, 
            'responses_length': responses_length,
            'kgs_h': np.array(kgs['h']),
            'kgs_r': np.array(kgs['r']),
            'kgs_t': np.array(kgs['t']),
            'kgs_h_length': np.array(kgs_length['h']),
            'kgs_r_length': np.array(kgs_length['r']),
            'kgs_t_length': np.array(kgs_length['t']),
            'match_triples': np.array(match_triples),
            'kgs_h_index': np.array(kgs_length['h']),
            'kgs_h_index_ga': np.array(kgs['h_index_ga']),
            'kgs_head_r': np.array(kgs['head_r']),
            'kgs_head_r_mask': np.array(kgs['head_r_mask']),
            }
    return batched_data

def train(model, sess, data_train):
    selected_data = [random.choice(data_train) for i in range(FLAGS.batch_size)]
    batched_data = gen_batched_data(selected_data)
    outputs = model.step_decoder(sess, batched_data)
    return outputs[0]

def evaluate(model, sess, data_dev, name='dev'):
    loss = np.zeros((1, ))
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(data_dev):
        selected_data = data_dev[st:ed]
        batched_data = gen_batched_data(selected_data)
        outputs = model.step_decoder(sess, batched_data, forward_only=True)
        loss += np.sum(outputs[0])
        st, ed = ed, ed+FLAGS.batch_size
    loss /= len(data_dev)
    print('perplexity on %s set: %.2f' % (name, np.exp(loss)))
    return np.exp(loss)[0]

def inference_set(sess, model, data_test, sample_num=1):
    data_dev = data_test
    if sample_num > 1:
        data_dev = [item for item in data_test for _ in range(sample_num)]
    st, ed = 0, FLAGS.batch_size
    responses = []
    while st < len(data_dev):
        selected_data = data_dev[st:ed]
        batched_data = gen_batched_data(selected_data)
        outputs = model.step_decoder(sess, batched_data, inference=True)
        responses += outputs[0].tolist()
        st, ed = ed, ed+FLAGS.batch_size
    results = []
    for response in responses:
        result = []
        for token in response:
            if type(token) is int:
                token = str(token)
            else:
                token = token.decode('utf8')
            if token != '_EOS':
                result.append(token)
            else:
                break
        if FLAGS.post_process:
            results.append(post_process(result))
        else:
            results.append(result)

    return results

def get_stat(data, responses):
    ngrams = {1: set(), 2: set(), 3: set(), 4: set()}
    ent_nums = []
    precisions, recalls, f1s = [], [], []
    num_ngrams = {1: 0, 2: 0, 3: 0, 4: 0}
    for i, response in enumerate(responses):
        golden_response = data[i]['response']
        ents = set()
        for triples in data[i]['kg']:
            for (h, r, t) in triples:
                ents.add(h[0])
                ents.add(t[0])

        golden_response_ents = set([word for word in set(data[i]['response']) if word in data[i]['entity2idx'] or word in data[i]['head_ents']])
        response_ents = set([word for word in response if word in data[i]['entity2idx'] or word in data[i]['head_ents']])
        golden_response_ents = set([word for word in set(data[i]['response']) if word in ents])
        response_ents = set([word for word in response if word in ents])
        union_ents = response_ents & golden_response_ents
        if len(golden_response_ents):
            precisions += [len(union_ents) / len(response_ents) if len(response_ents) else 0]
            recalls += [len(union_ents) / len(golden_response_ents)]
            f1s += [2 * precisions[-1] * recalls[-1] / (precisions[-1] + recalls[-1]) if (precisions[-1] + recalls[-1]) else 0]
        else:
            precisions += [-1]
            recalls += [-1]
            f1s += [-1]
        response_words = set(response)
        ent_nums += [sum([1 if word in ents else 0 for word in response_words])]
        for n in ngrams:
            for j in range(0, len(response)):
                ngram = tuple(response[j:j+n])
                if len(ngram) == n:
                    ngrams[n].add(ngram)
                    num_ngrams[n] += 1
    distinct = [0] * 5
    for n in ngrams:
        distinct[n] = len(ngrams[n]) / float(num_ngrams[n])
    distinct[0] = num_ngrams[1] / float(len(responses))
    return distinct[3:], ent_nums, precisions, recalls, f1s

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    if FLAGS.is_train:

        data_train, data_valid, data_test_seen, data_test_unseen = load_data(FLAGS.data_dir, 'data.json')
        vocab, embed = build_vocab(FLAGS.data_dir + '/vector.txt', data_train)
        data_train = [pair for session in data_train for pair in session]
        data_valid = [pair for session in data_valid for pair in session]
        data_test_seen = [pair for session in data_test_seen for pair in session]
        data_test_unseen = [pair for session in data_test_unseen for pair in session]
        
        model = Seq2SeqModel(
                FLAGS.symbols, 
                FLAGS.embed_units,
                FLAGS.units, 
                FLAGS.layers,
                FLAGS.beam_size,
                embed,
                )
        if FLAGS.log_parameters:
            print(FLAGS.flag_values_dict())
            model.print_parameters()
        
        if tf.train.get_checkpoint_state(FLAGS.train_dir):
            print("Reading model parameters from %s" % FLAGS.train_dir)
            tf.global_variables_initializer().run()
            model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        else:
            print("Created model with fresh parameters.")
            tf.global_variables_initializer().run()
            op_in = model.symbol2index.insert(constant_op.constant(vocab),
                constant_op.constant(list(range(FLAGS.symbols)), dtype=tf.int64))
            sess.run(op_in)
            op_out = model.index2symbol.insert(constant_op.constant(
                list(range(FLAGS.symbols)), dtype=tf.int64), constant_op.constant(vocab))
            sess.run(op_out)

        loss_step, time_step = np.zeros((1, )), .0
        previous_losses = [1e18]*3
        while True:
            if model.global_step.eval() % FLAGS.per_checkpoint == 0:
                show = lambda a: '[%s]' % (' '.join(['%.2f' % x for x in a]))
                print("global step %d learning rate %.6f step-time %.6f perplexity %s"
                        % (model.global_step.eval(), model.learning_rate.eval(), 
                            time_step, show(np.exp(loss_step))))
                model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, 
                        global_step=model.global_step)
                ppx = evaluate(model, sess, data_valid, name='valid')
                if ppx < model.ppx_best.eval():
                    sess.run(model.ppx_best.assign(ppx))
                    model.saver_best.save(sess, '%s/best/checkpoint' % FLAGS.train_dir, 
                            global_step=model.global_step)
                if np.sum(loss_step) > max(previous_losses):
                    sess.run(model.learning_rate_decay_op)
                previous_losses = previous_losses[1:]+[np.sum(loss_step)]
                loss_step, time_step = np.zeros((1, )), .0
                sys.stdout.flush()
                if model.global_step.eval() == 20000:
                    exit()

            start_time = time.time()
            loss_step += train(model, sess, data_train) / FLAGS.per_checkpoint
            time_step += (time.time() - start_time) / FLAGS.per_checkpoint
                
    else:
        model = Seq2SeqModel(
                FLAGS.symbols, 
                FLAGS.embed_units,
                FLAGS.units, 
                FLAGS.layers,
                FLAGS.beam_size,
                embed=None,
                )

        if FLAGS.log_parameters:
            print(FLAGS.flag_values_dict())
            model.print_parameters()
            sys.stdout.flush()

        if FLAGS.inference_version == 0:
            model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        else:
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)
        print('restore from %s' % model_path)
        model.saver.restore(sess, model_path)

        data_train, data_valid, data_test_seen, data_test_unseen = load_data(FLAGS.data_dir, 'data.json')
        data_train = [pair for session in data_train for pair in session]
        data_valid = [pair for session in data_valid for pair in session]
        data_test_seen = [pair for session in data_test_seen for pair in session]
        data_test_unseen = [pair for session in data_test_unseen for pair in session]

        evaluate(model, sess, data_valid, name='valid')
        evaluate(model, sess, data_test_seen, name='test_seen')
        evaluate(model, sess, data_test_unseen, name='test_unseen')

        data_test = data_test_seen + data_test_unseen
        posts = [pair['post'] for pair in data_test]
        responses = []
        st, ed = 0, FLAGS.batch_size

        responses_seen = inference_set(sess, model, data_test_seen)
        distinct, ent_nums, precisions, recalls, f1s = get_stat(data_test_seen, responses_seen)
        print(('\ninference test_seen:\ndistinct-3: %f\ndistinct-4: %f\nent_num: %f\nprecision: %f\nrecall" %f\nf1: %f\n') % 
                tuple(distinct + [np.mean(ent_nums), np.mean([num for num in precisions if num > -1]), 
                np.mean([num for num in recalls if num > -1]), np.mean([num for num in f1s if num > -1])]))

        responses_unseen = inference_set(sess, model, data_test_unseen)
        distinct, ent_nums, precisions, recalls, f1s = get_stat(data_test_unseen, responses_unseen)
        print(('inference test_unseen:\ndistinct-3: %f\ndistinct-4: %f\nent_num: %f\nprecision: %f\nrecall" %f\nf1: %f\n') % 
                tuple(distinct + [np.mean(ent_nums), np.mean([num for num in precisions if num > -1]), 
                np.mean([num for num in recalls if num > -1]), np.mean([num for num in f1s if num > -1])]))

        responses = responses_seen + responses_unseen

        with open(FLAGS.inference_path+'.out', 'w', encoding='utf8') as f:
            for i in range(len(responses)):
                f.write('post: %s\nresponse: %s\ngolden: %s\n\n' 
                    % (' '.join(posts[i]), ' '.join(responses[i]), ' '.join(data_test[i]['response'])))
