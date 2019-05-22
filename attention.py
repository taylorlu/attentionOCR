import tensorflow as tf
import backbone


class AttentionModel(object):

    def __init__(self, args):
        self.init_learning_rate = args.get('learning_rate', 0.002)
        self.max_grad_norm = args.get('max_grad_norm', 100)
        self.decay_steps = args.get('decay_steps', 500)
        self.decay_rate = args.get('decay_rate', 0.95)
        self.embedding_dim = args.get('embedding_dim', 256)
        self.vocabulary_size = args.get('vocabulary_size', 3582)
        self.max_step = args.get('max_step', 50)
        self.num_lstm_units = args.get('num_lstm_units', 256)
        self.batch_size = args.get('batch_size', 32)
        self.attention_loss_factor = 0.01
        self.l2_regularizer = tf.contrib.layers.l2_regularizer(1e-4)

        self._init_inference = False
        self._init_cost = False
        self._init_train = False


    def initialize(self, context_mean):
        """ Initialize the LSTM using the mean context. """

        memory = tf.layers.dense(context_mean, units=self.num_lstm_units)
        output = tf.layers.dense(context_mean, units=self.num_lstm_units)
        return memory, output


    def attend(self, contexts, output, attention_dim=256, is_train=True):
        # contexts(batch, w*h, d), output(batch, h)

        reshaped_contexts = tf.reshape(contexts, [-1, self.channels])  # (batch*w*h, d)

        reshaped_contexts = tf.layers.dropout(reshaped_contexts, rate=0.0)
        output = tf.layers.dropout(output, rate=0.0)    # (batch, h)

        temp1 = tf.layers.dense(reshaped_contexts, units=attention_dim) # (batch*w*h, att)
        temp2 = tf.layers.dense(output, units=attention_dim)
        temp2 = tf.tile(tf.expand_dims(temp2, 1), [1, self.num_ctx, 1])
        temp2 = tf.reshape(temp2, [-1, attention_dim])  # (batch*w*h, att)

        temp = tf.math.tanh(temp1 + temp2)    # (batch*w*h, att)

        temp = tf.layers.dropout(temp, rate=0.0)

        logits = tf.layers.dense(temp, units=1, use_bias=False) # (batch*w*h, 1)

        logits = tf.reshape(logits, [-1, self.num_ctx])  # (batch, w*h)
        alpha = tf.nn.softmax(logits)     # (batch, w*h)

        return alpha


    def buildAttention(self, features, labels=None, masks=None,
                        last_word=None, last_output=None, last_memory=None,
                        is_train=True):

        features = tf.reshape(features, [self.batch_size, -1, self.channels])  # (batch, w*h, d)
        # Batch Normalization
        bn_op = tf.keras.layers.BatchNormalization(axis=-1, name='bn')
        features = bn_op(features, training=is_train)
        if(is_train):
            for operation in bn_op.updates:
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, operation)

        with tf.variable_scope("word_embedding"):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [self.vocabulary_size, self.embedding_dim],
                initializer = tf.contrib.layers.xavier_initializer(),
                regularizer = self.l2_regularizer,
                trainable = is_train)

        with tf.variable_scope("decode"):
            weight_context = tf.get_variable(
                name = 'weight_context',
                shape = [self.channels, self.embedding_dim],
                initializer = tf.contrib.layers.xavier_initializer(),
                regularizer = self.l2_regularizer,
                trainable = is_train)
            weight_out = tf.get_variable(
                name = 'weight_out',
                shape = [self.num_lstm_units, self.embedding_dim],
                initializer = tf.contrib.layers.xavier_initializer(),
                regularizer = self.l2_regularizer,
                trainable = is_train)

        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.num_lstm_units, initializer=tf.orthogonal_initializer())
        if(is_train):
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=0.75, output_keep_prob=0.75)

        # Initialize the LSTM using the mean context
        with tf.variable_scope("initialize"):
            context_mean = tf.reduce_mean(features, axis=1)
            self.initial_memory, self.initial_output = self.initialize(context_mean)

            # Training only initial once, Testing stage need last state as input
            if(is_train):
                last_memory = self.initial_memory
                last_output = self.initial_output
            last_state = last_memory, last_output

        predictions = []
        if(is_train):
            last_word = tf.zeros([self.batch_size], tf.int32)
            cross_entropies = []
            alphas = []
        else:
            self.max_step = 1

        for idx in range(self.max_step):
            # Attention mechanism
            with tf.variable_scope("attend", reuse=tf.AUTO_REUSE):
                alpha = self.attend(features, last_output, True)  # (batch, w*h)
                context = tf.reduce_sum(features*tf.expand_dims(alpha, 2), axis=1)  # (batch, d)
                if(is_train):
                    tiled_masks = tf.tile(tf.expand_dims(masks[:, idx], 1),
                                         [1, self.num_ctx])
                    masked_alpha = alpha * tiled_masks
                    alphas.append(tf.reshape(masked_alpha, [-1]))

            # Embed the last word
            with tf.variable_scope("word_embedding", reuse=tf.AUTO_REUSE):
                word_embed = tf.nn.embedding_lookup(embedding_matrix, last_word)    # (batch, emb)

            with tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):
                current_input = tf.concat([context, word_embed, last_output], 1)
                output, state = lstm_cell(current_input, last_state)

            # Decode the expanded output of LSTM into a word
            with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):
                context_logits = tf.matmul(context, weight_context)
                output_logits = tf.matmul(output, weight_out)
                expanded_output = context_logits +output_logits +word_embed # (batch, emb)

                if(is_train):
                    expanded_output = tf.layers.dropout(expanded_output, rate=0.5)

                logits = tf.layers.dense(expanded_output, units=self.vocabulary_size, use_bias=False)
                probs = tf.nn.softmax(logits)
                prediction = tf.argmax(logits, 1)
                predictions.append(prediction)

            # Compute the loss for this step, if necessary
            if(is_train):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels[:, idx], logits=logits)
                masked_cross_entropy = cross_entropy * masks[:, idx]
                cross_entropies.append(masked_cross_entropy)
                last_word = labels[:, idx]

            # Current step state
            last_output = output
            last_state = state
            last_memory = state[1]

        if(is_train):
            cross_entropies = tf.stack(cross_entropies, axis=1)
            cross_entropy_loss = tf.reduce_sum(cross_entropies) / tf.reduce_sum(masks)

            alphas = tf.stack(alphas, axis=1)
            alphas = tf.reshape(alphas, [self.batch_size, self.num_ctx, -1])
            attentions = tf.reduce_sum(alphas, axis=2)
            diffs = tf.ones_like(attentions) - attentions
            attention_loss = self.attention_loss_factor \
                             * tf.nn.l2_loss(diffs) \
                             / (self.batch_size * self.num_ctx)

            reg_loss = tf.losses.get_regularization_loss()
            total_loss = cross_entropy_loss + attention_loss + reg_loss
            return total_loss
        else:
            return probs, last_output, last_memory


    def init_inference(self):
        # feed inputs placeholder here
        self.inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], name='inputs')
        self.last_word = tf.placeholder(tf.int32, [None], name='last_word')
        self.last_output = tf.placeholder(tf.float32, [None, self.num_lstm_units], name='last_output')
        self.last_memory = tf.placeholder(tf.float32, [None, self.num_lstm_units], name='last_memory')

        features = backbone.resnet_2D_v1(self.inputs, trainable=False)
        self.channels = features.get_shape().as_list()[-1]
        self.num_ctx = features.get_shape().as_list()[1]*features.get_shape().as_list()[2]
        probs, last_output, last_memory = self.buildAttention(features, last_word=self.last_word,
                                                                last_output=self.last_output,
                                                                last_memory=self.last_memory,
                                                                is_train=False)
        return probs, last_output, last_memory


    def init_inference_for_train(self):
        # ===============================================
        #           build network and loss
        # ===============================================
        # feed inputs, labels and masks placeholder here
        self.inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input')
        self.labels = tf.placeholder(tf.int32, [None, self.max_step], name='label')
        self.masks = tf.placeholder(tf.float32, [None, self.max_step], name='mask')

        features = backbone.resnet_2D_v1(self.inputs, trainable=True)
        self.channels = features.get_shape().as_list()[-1]
        self.num_ctx = features.get_shape().as_list()[1]*features.get_shape().as_list()[2]
        self._cost = self.buildAttention(features, self.labels, self.masks, is_train=True)
        self._init_cost = True


    def init_train(self, train_vars=None):
        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        self._lr = tf.train.exponential_decay(self.init_learning_rate, self._global_step,
                    self.decay_steps, self.decay_rate, staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(self._lr)
            grads, tvars = zip(*optimizer.compute_gradients(self._cost, train_vars))
            grads_clip, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            self._train_op = optimizer.apply_gradients(zip(grads_clip, tvars), global_step=self._global_step)
        self._init_train = True


    def feed_dict(self, inputs, labels=None, masks=None,
                    last_word=None, last_output=None, last_memory=None):
        """
        Constructs the feed dictionary from given inputs necessary to run
        an operations for the model.

        Args:
            inputs : 4D numpy array feature map. Should be
                of shape [batch, height, width, channels]
            labels : List of labels for each item in the batch. Each label
                should be a list of integers. If label=None does not feed the
                label placeholder (for e.g. inference only).

        Returns:
            A dictionary of placeholder keys and feed values.
        """
        feed_dict = {self.inputs : inputs}

        if(labels is not None):
            label_dict = {self.labels : labels}
            feed_dict.update(label_dict)
        if(masks is not None):
            mask_dict = {self.masks : masks}
            feed_dict.update(mask_dict)

        if(last_word is not None):
            last_word_dict = {self.last_word : last_word}
            feed_dict.update(last_word_dict)
        if(last_output is not None):
            last_output_dict = {self.last_output : last_output}
            feed_dict.update(last_output_dict)
        if(last_memory is not None):
            last_memory_dict = {self.last_memory : last_memory}
            feed_dict.update(last_memory_dict)

        return feed_dict



    @property
    def cost(self):
        assert self._init_cost, "Must init inference_for_train module."
        return self._cost

    @property
    def train_op(self):
        assert self._init_train, "Must init train module."
        return self._train_op

    @property
    def global_step(self):
        assert self._init_train, "Must init train module."
        return self._global_step

    @property
    def learning_rate(self):
        assert self._init_train, "Must init train module."
        return self._lr



def main():
    features = tf.placeholder(shape=[32, 100,10, 64], dtype=tf.float32)
    labels = tf.placeholder(shape=[32, 2000], dtype=tf.int32)
    masks = tf.placeholder(shape=[32, 60], dtype=tf.int32)
    buildAttention(features, labels, masks)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, 'ckpt/model')


if(__name__=='__main__'):
    main()
