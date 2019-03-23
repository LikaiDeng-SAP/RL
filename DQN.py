import numpy as np
import tensorflow as tf
import os

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.1,
            reward_decay=0.9,
            e_greedy_start=0.6,
            replace_target_iter=10,
            memory_size=500,
            batch_size=64,
            e_greedy_final=0.0,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_final = e_greedy_final
        self.epsilon = e_greedy_start

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        self.saver = tf.train.Saver()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        # 原始版本
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            f1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='f1')

            f2 = tf.layers.dense(f1, 32, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='f2')

            self.q_eval = tf.layers.dense(f2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        self.y_ = tf.placeholder(tf.float32, [None, self.n_actions])

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_eval, self.y_, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1


    def choose_action(self, observation):

        observation = observation[np.newaxis, :]

        if np.random.uniform() >= self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        state_minibatch = []
        y_minibatch = []
        minibatch_size = min(len(self.memory), self.batch_size)
        minibatch_indexes = np.random.randint(0, len(self.memory), minibatch_size)

        for j in minibatch_indexes:
            state_j = self.memory[j][:self.n_features]
            action_j = self.memory[j][self.n_features]
            reward_j= self.memory[j][self.n_features+1]
            state_j_1 = self.memory[j][-self.n_features:]

            y_j = self.sess.run(self.q_eval,feed_dict={self.s:np.reshape(state_j,(-1,self.n_features))})

            if reward_j == -1:
                y_j[0][int(action_j)] = reward_j
            else:
                # reward_j + gamma * max_action' Q(state', action')
                y_j[0][int(action_j)] = reward_j + self.gamma * np.max(self.sess.run(self.q_eval,feed_dict={self.s:np.reshape(state_j_1,(-1,self.n_features))}))  # NOQA
            state_minibatch.append(state_j)
            y_minibatch.append(y_j[0])

        # training
        self.sess.run(self._train_op, feed_dict={self.s: state_minibatch, self.y_: y_minibatch})

        # decreasing epsilon
        self.epsilon = self.epsilon - 0.0005 if self.epsilon > self.epsilon_final else self.epsilon

        self.learn_step_counter += 1
        #print(self.learn_step_counter)

    def save_model(self):
        self.saver.save(self.sess, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"), "{}.ckpt".format('uav_circle')))

    def load_model(self):
        checkpoint = tf.train.get_checkpoint_state('models')
        self.saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
        self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
        print('Load model!')