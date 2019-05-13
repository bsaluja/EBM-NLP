import numpy as np
import os
import tensorflow as tf

import sys
sys.path.append('../')
import eval

from .data_utils import minibatches, pad_sequences, get_chunks, NONE
from .general_utils import Progbar
from .base_model import BaseModel

from .visualization_utils import plot_confusion_graph, plot_bar_graph, plot_bar_graph_for_two_data_series

import string

class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           list(self.config.vocab_tags.items())}
        self.tag_to_idx = {tag: idx for tag, idx in
                           list(self.config.vocab_tags.items())}


    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = list(zip(*words))
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths


    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], 2*self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)


    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])


    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                    tf.int32)


    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths


    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics, labels_dict = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in list(metrics.items())])
        self.logger.info(msg)

        return metrics["f1"]

    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        def div_or_zero(num, den):
          return num/den if den else 0.0

        l_true = []
        l_pred = []

        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.

        #Skipping punctuations while evaluation of state-of-the-art model
        # punctuations = string.punctuation
        # totalPunctuationsCount = 0
        totalGoldConditionSpansCount = 0
        totalPredictedConditionSpansCount = 0
        totalConditionSpansIntersectionCount = 0
        totalConditionSpansExactMatchCount = 0
        abstractNum = 0

        #Visualization Stats
        goldConditionSpansCountAbstractWise = []
        predictedConditionSpansCountAbstractWise = []
        abstractsNumList = []

        #((abstractNum, numGoldConditionSpansPerAbstract, numPredictedConditionSpansPerAbstract, numConditionIntersectionsGoldPredicted, numConditionExactMatchGoldPredicted, goldConditionSpansPerAbstract, predictedConditionSpansPerAbstract, conditionIntersectionList, conditionExactMatchList))       
        final_condition_span_stats = []

        #Skipping punctuations while evaluation of state-of-the-art model
        # punctuations_processed = []
        # punctuations_vocab = {}
        # for punct in punctuations:
        #     punct_processed = self.config.processing_word(punct)
        #     punctuations_processed.append(punct_processed)
        #     punctuations_vocab[punct] = punct_processed

        # if not os.path.exists(self.config.dir_punctuations):
        #     os.makedirs(self.config.dir_punctuations)

        # filename_punctuations_vocab = os.path.join(self.config.dir_punctuations, "punct_vocab.txt")
        # with open(filename_punctuations_vocab, "w") as f:
        #     for i, (key, value) in enumerate(punctuations_vocab.items()):
        #         if i != len(punctuations_vocab) - 1:
        #             f.write("{} => {}\n".format(value,key))
        #         else:
        #             f.write("{} => {}".format(value,key))

        for words, labels in minibatches(test, self.config.batch_size):

            # BS - Begin
            abstractNum = abstractNum + 1
            abstractsNumList.append(abstractNum)
            print('Evaluating Abstract #%d:' %(abstractNum))

            # numPunctuationsPerAbstract = 0
            # punctuationsPerAbstract = {}
            numGoldConditionSpansPerAbstract = 0
            numPredictedConditionSpansPerAbstract = 0
            goldConditionSpansPerAbstract = []          #[[span1 indices], [span2 indices]]
            predictedConditionSpansPerAbstract = []     #[[span1 indices], [span2 indices]]
            
            # Skipping punctuations while evaluation of state-of-the-art model
            # words_without_punctuation = []
            # unzipped = []
            # for (a,b) in words:
            #     for (char_ids, word_id) in zip(a, b):
            #         unzipped += [(char_ids, word_id)]

            # for index, (char_ids, word_id) in enumerate(unzipped):
            #     if (char_ids, word_id) in punctuations_processed:
            #         totalPunctuationsCount = totalPunctuationsCount + 1
            #         numPunctuationsPerAbstract = numPunctuationsPerAbstract + 1
            #         punctuationsPerAbstract[index] = ((char_ids, word_id))
            #         # print("Punctuation Found:: punct_char_id = " +  str(char_ids) + "; punct_word_id = " + str(word_id))
            #     else:
            #         words_without_punctuation += [(char_ids, word_id)]

            # if type(words_without_punctuation[0]) == tuple:
            #     words = [zip(*words_without_punctuation)]

            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):

                lab      = lab[:length]
                lab_pred = lab_pred[:length]

                goldConditionSpansPerAbstract, predictedConditionSpansPerAbstract = self.count_consecutive_condition_labels(lab, lab_pred)
                numGoldConditionSpansPerAbstract = len(goldConditionSpansPerAbstract)
                numPredictedConditionSpansPerAbstract = len(predictedConditionSpansPerAbstract)

                totalGoldConditionSpansCount += numGoldConditionSpansPerAbstract
                totalPredictedConditionSpansCount += numPredictedConditionSpansPerAbstract
                goldConditionSpansCountAbstractWise.append(numGoldConditionSpansPerAbstract)
                predictedConditionSpansCountAbstractWise.append(numPredictedConditionSpansPerAbstract)
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                l_true += lab
                l_pred += lab_pred

                # print('Punctuations Count for Abstract #%d = %d' %(abstractNum, numPunctuationsPerAbstract))
                # if not os.path.exists(self.config.dir_punctuations):
                #     os.makedirs(self.config.dir_punctuations)

                # filename_punctuations = os.path.join(self.config.dir_punctuations, "punctuationsAbstract{}.txt".format(abstractNum))
                # with open(filename_punctuations, "w") as f:
                #     for i, (index, punctuation) in enumerate(punctuationsPerAbstract.items()):
                #         if i != len(punctuationsPerAbstract) - 1:
                #             f.write("{}\t{}\n".format(index, punctuation))
                #         else:
                #             f.write("{}\t{}".format(index, punctuation))

                # partly match (intersection)
                conditionIntersectionList = self.intersection_gold_predict_condition_spans(goldConditionSpansPerAbstract, predictedConditionSpansPerAbstract)
                numConditionIntersectionsGoldPredicted = len(conditionIntersectionList)
                totalConditionSpansIntersectionCount += numConditionIntersectionsGoldPredicted

                # entirely exact match
                conditionExactMatchList = self.exact_match_gold_predict_condition_spans(goldConditionSpansPerAbstract, predictedConditionSpansPerAbstract)
                numConditionExactMatchGoldPredicted = len(conditionExactMatchList)
                totalConditionSpansExactMatchCount += numConditionExactMatchGoldPredicted

                # (Abstract#, numGoldConditionSpansPerAbstract, goldConditionSpansPerAbstract, numPredictedConditionSpansPerAbstract, predictedConditionSpansPerAbstract)
                final_condition_span_stats.append((abstractNum, numGoldConditionSpansPerAbstract, numPredictedConditionSpansPerAbstract, numConditionIntersectionsGoldPredicted, numConditionExactMatchGoldPredicted, goldConditionSpansPerAbstract, predictedConditionSpansPerAbstract, conditionIntersectionList, conditionExactMatchList))

        if not os.path.exists(self.config.dir_conditon_spans_results):
            os.makedirs(self.config.dir_conditon_spans_results)
        # (Abstract#, numGoldConditionSpansPerAbstract, goldConditionSpansPerAbstract, numPredictedConditionSpansPerAbstract, predictedConditionSpansPerAbstract)
        filename_condition_spans = os.path.join(self.config.dir_conditon_spans_results, "conditionSpanMetrics.txt")
        with open(filename_condition_spans, "w") as f:
            # f.write("Abstract#\tnumGoldConditionSpansPerAbstract\tnumPredictedConditionSpansPerAbstract\tnumConditionIntersectionsGoldPredicted\tnumConditionExactMatchGoldPredicted\tgoldConditionSpansPerAbstract\tpredictedConditionSpansPerAbstract\tconditionIntersectionList\tconditionExactMatchList\n")
            f.write("Abstract#\tGold#\tPredicted#\tPartlyMatch#\tExactMatch#\tPartlyMatch%\tExactMatch%\tGoldSpan\tPredictedSpan\tPartlyMatchSpan\tExactMatchSpan\n")
            for i, _tuple in enumerate(final_condition_span_stats):
                # partlyMatchPerc = ((_tuple[3]/_tuple[1])*100)
                # exactMatchPerc = ((_tuple[4]/_tuple[1])*100)
                partlyMatchPerc = str(div_or_zero(_tuple[3], _tuple[1]) * 100) + "%"
                exactMatchPerc = str(div_or_zero(_tuple[4], _tuple[1]) * 100) + "%"

            # if i != len(final_condition_span_stats) - 1:
                f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(_tuple[0], _tuple[1], _tuple[2], _tuple[3], _tuple[4], partlyMatchPerc, exactMatchPerc, _tuple[5], _tuple[6], _tuple[7], _tuple[8]))
            #     else:
            #         f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(_tuple[0], _tuple[1], _tuple[2], _tuple[3], _tuple[4], ((_tuple[3]/_tuple[1])*100), ((_tuple[4]/_tuple[1])*100), _tuple[5], _tuple[6], _tuple[7], _tuple[8]))
            totalPartlyMatchPerc = str(div_or_zero(totalConditionSpansIntersectionCount, totalGoldConditionSpansCount) *100) + "%"
            totalExactMatchPerc = str(div_or_zero(totalConditionSpansExactMatchCount, totalGoldConditionSpansCount) *100) + "%"
            f.write("Total\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(totalGoldConditionSpansCount, totalPredictedConditionSpansCount, totalConditionSpansIntersectionCount, totalConditionSpansExactMatchCount, totalPartlyMatchPerc, totalExactMatchPerc, "NA", "NA", "NA", "NA"))

        if not os.path.exists(self.config.dir_graphics):
            os.makedirs(self.config.dir_graphics)

        filename_for_fig = os.path.join(self.config.dir_graphics, "conditionspanabstract_barplot.png")

        plot_bar_graph_for_two_data_series(abstractsNumList, goldConditionSpansCountAbstractWise, predictedConditionSpansCountAbstractWise, 'Gold', 'Predicted', 'Abstracts', 'Number of Condition Spans', 'Number of Condition Spans Per Abstract', filename_for_fig)

        if not os.path.exists(self.config.dir_counts_eval_metrics):
            os.makedirs(self.config.dir_counts_eval_metrics)
        filename_counts_eval_metrics = os.path.join(self.config.dir_counts_eval_metrics, "counts_eval_metrics.txt")
        with open(filename_counts_eval_metrics, "w") as f:
            # f.write("totalPunctuationsCount = {}\n".format(totalPunctuationsCount))
            f.write("totalGoldConditionSpansCount = {}\n".format(totalGoldConditionSpansCount))
            f.write("totalPredictedConditionSpansCount = {}\n".format(totalPredictedConditionSpansCount))

        # Token stats
        print('Passing LSTM-CRF tags to eval func:')
        print('\t', self.idx_to_tag.items())
        tags = [idx for idx, tag in self.idx_to_tag.items() if tag != NONE]
        return eval.token_f1(true = l_true, pred = l_pred, labels = tags), self.idx_to_tag.items()


    def intersection(self, lst1, lst2):
      temp = set(lst2)
      lst3 = [value for value in lst1 if value in temp]
      return lst3

    def exact_match(self, lst1, lst2):
        all_equal = True
        for ele_lst1 in lst1:
            for ele_lst2 in lst2:
                if ele_lst1 != ele_lst2:
                    all_equal = False
                    break

        if all_equal == True:
            return lst1

        return []

    def intersection_gold_predict_condition_spans(self, goldSpansList, predictSpansList):
        intersectListFinal = []
        for goldSpan in goldSpansList:
            for predictedSpan in predictSpansList:
                curr_intersectList = self.intersection(goldSpan, predictedSpan)
                if curr_intersectList:
                    intersectListFinal.append(curr_intersectList)

        return intersectListFinal

    def exact_match_gold_predict_condition_spans(self, goldSpansList, predictSpansList):
        exact_match_final_list = []
        for goldSpan in goldSpansList:
            for predictedSpan in predictSpansList:
                curr_exact_match_list = self.exact_match(goldSpan, predictedSpan)
                if curr_exact_match_list:
                    exact_match_final_list.append(curr_exact_match_list)

        return exact_match_final_list

    def count_consecutive_condition_labels(self, lab, lab_pred):
        tag_idx_for_condition = self.tag_to_idx['4_p']
        gold_condition_spans_indices = []
        predicted_condition_spans_indices = []

        current_gold_condition_span = []
        current_predicted_condition_span = []
        for idx, (gold_label, predicted_label) in enumerate(zip(lab, lab_pred)):
            one_based_idx = idx + 1
            if gold_label == tag_idx_for_condition:
                current_gold_condition_span.append(one_based_idx)
            else:
                if current_gold_condition_span:
                    gold_condition_spans_indices.append(current_gold_condition_span)
                    current_gold_condition_span = []

            if predicted_label == tag_idx_for_condition:
                current_predicted_condition_span.append(one_based_idx)
            else:
                if current_predicted_condition_span:
                    predicted_condition_spans_indices.append(current_predicted_condition_span)
                    current_predicted_condition_span = []

        # print("GoldConditionSpansIndices = ", gold_condition_spans_indices)
        # print("PredictedConditionSpansIndices = ", predicted_condition_spans_indices)

        # print("NumGoldConditionSpans = ", len(gold_condition_spans_indices))
        # print("NumPredictedConditionSpans = ", len(predicted_condition_spans_indices))

        return gold_condition_spans_indices, predicted_condition_spans_indices

    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = list(zip(*words))
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds