import numpy as np
import tensorflow as tf
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import tensorflow.contrib.layers as layers
# import os
# os.environ[ "CUDA_VISIBLE_DEVICES"] = "-1"

#build DQN_agent
class DeepQNetwork():
    def __init__(self, learning_rate, state_size, action_size, step_size,
                 hidden_size, sess=None, gamma=0.9, epsilon=0.8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_size = action_size
        self.state_size =  state_size
        self.step_size = step_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        #self.name = "q_network"
        self.alpha = 0  # co-operative fairness constant
        self.beta = 1  # Annealing constant for Monte - Carlo
        self.network()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.write = tf.summary.FileWriter("DRQN/summaries", sess.graph)


    def network(self):
        #q_net
        scope_var = "q_network"
        with tf.variable_scope(scope_var, reuse=False):
            self.inputs_q = tf.placeholder(tf.float32, [None, self.step_size, self.state_size], name='inputs_q')
            lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, name='q_lstm' )
            lstm_out, state = tf.nn.dynamic_rnn(lstm, self.inputs_q, dtype=tf.float32)
            reduced_out = lstm_out[:, -1, :]
            reduced_out = tf.reshape(reduced_out, shape=[-1, self.hidden_size])

            w2 = tf.Variable(tf.random_uniform([self.hidden_size, self.hidden_size]))
            b2 = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]))
            h2 = tf.matmul(reduced_out, w2) + b2
            h2 = tf.nn.relu(h2, 'relu_q')
            h2 = tf.contrib.layers.layer_norm(h2, scope='q_norm')

            out_v = tf.contrib.layers.fully_connected(h2, num_outputs=1, activation_fn=None, scope='q_full_V')
            out_a = tf.contrib.layers.fully_connected(h2, num_outputs=self.action_size, activation_fn=None, scope='q_full_A')
            self.q_value = out_v + (out_a - tf.reduce_mean(out_a, axis=1, keepdims=True))

        #target_net
        scope_tar = "target_network"
        with tf.variable_scope(scope_var, reuse=False):
            self.inputs_target = tf.placeholder(tf.float32, [None, self.step_size, self.state_size], name='inputs_target')
            lstm_t = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, name='t_lstm')
            lstm_out_t, state_t = tf.nn.dynamic_rnn(lstm_t, self.inputs_target, dtype=tf.float32)
            reduced_out_t = lstm_out_t[:, -1, :]
            reduced_out_t = tf.reshape(reduced_out_t, shape=[-1, self.hidden_size])

            w2_t = tf.Variable(tf.random_uniform([self.hidden_size, self.hidden_size]))
            b2_t = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]))
            h2_t = tf.matmul(reduced_out_t, w2_t) + b2_t
            h2_t = tf.nn.relu(h2_t, 'relu_t')
            h2_t = tf.contrib.layers.layer_norm(h2_t, scope= 't_norm')

            out_v_t = tf.contrib.layers.fully_connected(h2_t, num_outputs=1, activation_fn=None, scope='t_full_V')
            out_a_t = tf.contrib.layers.fully_connected(h2_t, num_outputs=self.action_size, activation_fn=None, scope='t_full_A')
            self.q_target = out_v_t + (out_a_t - tf.reduce_mean(out_a_t, axis=1, keepdims=True))

        with tf.variable_scope("loss"):
            self.actions = tf.placeholder(tf.int32, [None], name='actions')
            with tf.device('/cpu:0'):
                one_hot_actions = tf.one_hot(self.actions, self.action_size)
            Q = tf.reduce_sum(tf.multiply(self.q_value, one_hot_actions), axis=1)
            self.target = tf.placeholder(tf.float32, [None], name='target')
            self.loss = tf.reduce_mean(tf.square(self.target - Q))
            tf.summary.scalar('loss', tf.reduce_mean(self.loss))

        with tf.variable_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def train(self, state, reward, action, state_next):
        q_target = self.sess.run(self.q_target, feed_dict={self.inputs_target: state_next})
        targets = reward + self.gamma * np.max(q_target, axis=1)
        # print(targets)
        # print(np.shape(action))
        # print(action)
        summery, loss, _ = self.sess.run([self.merged, self.loss, self.train_op],
                                         feed_dict={self.inputs_q: state, self.target: targets,
                                                    self.actions: action})
        return summery, loss

    def chose_action_train(self, current_state):
        #current_state = current_state[np.newaxis, :]  # *** array dim: (xx,)  --> (1 , xx) ***
        # print(np.shape(current_state))
        q = self.sess.run(self.q_value, feed_dict={self.inputs_q: current_state})
        #q = self.sess.run(self.out_q, feed_dict={self.inputs: current_state})
        # print(q)


        # e-greedy
        if np.random.random() < self.epsilon:
            action_chosen = np.random.randint(0, self.action_size)
        else:
            action_chosen = np.argmax(q, axis=1)
        # print(np.argmax(q))
        return action_chosen

    def chose_action_test(self, current_state):
        #current_state = current_state[np.newaxis, :]  # *** array dim: (xx,)  --> (1 , xx) ***
        q = self.sess.run(self.q_value, feed_dict={self.inputs_q: current_state})
        # print(q)

        action_chosen = np.argmax(q, axis=1)
        # print(np.argmax(q), q)
        return action_chosen

    # upadate parmerters
    def update_prmt(self):
        q_prmts = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "q_network")
        target_prmts = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "target_network")
        self.sess.run([tf.assign(t, q) for t, q in zip(target_prmts, q_prmts)])  # ***
        #print("updating target-network parmeters...")

    def decay_epsilon(self):
        if self.epsilon > 0.03:
            self.epsilon = self.epsilon - 0.02

#创建储存智能体
class ExpMemory():
    def __init__(self, in_size):
        self.buffer_in = deque(maxlen=in_size)

    def add(self, exp):
        self.buffer_in.append(exp)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer_in)), size=batch_size, replace=False)
        res = []
        for i in idx:
            res.append(self.buffer_in[i])
        return res


            #读取数据集

#创建actor-critic智能体
class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001, epsilon=0.8):
        self.sess = sess
        self.epsilon = epsilon
        self.n_actions=n_actions
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=200,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # 获取所有操作的概率

        # e-greedy
        if np.random.random() < self.epsilon:
            action_chosen = np.random.randint(0, self.n_actions)
        else:
            action_chosen = np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())
        return action_chosen  # return a int

    def decay_epsilon(self):
        if self.epsilon > 0.03:
            self.epsilon = self.epsilon - 0.02

class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=200,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_: v_, self.r: r})
        return td_error


#编写环境
#From one state,actions and observations make new states
def state_gen(state,action,obs):
    state_out = state.tolist()
    state_out.append(action)
    state_out.append(obs)
    state_out = state_out[2:]
    return np.asarray(state_out)

# Fetch states,actions,observations and next state from memory
def get_states(batch):
    states = []
    for i in batch:
        states.append(i[0])
    state_arr = np.asarray(states)
    state_arr = state_arr.reshape(32,32)
    #state_arr = state_arr.reshape(8, 32)
    return state_arr

def get_actions(batch):
    actions = []
    for i in batch:
        actions.append(i[1])
    actions_arr = np.asarray(actions)
    actions_arr = actions_arr.reshape(32)
    #actions_arr = actions_arr.reshape(8)
    return actions_arr

def get_rewards(batch):
    rewards = []
    for i in batch:
        rewards.append(i[2])
    rewards_arr = np.asarray(rewards)
    rewards_arr = rewards_arr.reshape(32)
    #rewards_arr = rewards_arr.reshape(8)
    return rewards_arr

def get_next_states(batch):
    next_states = []
    for i in batch:
        next_states.append(i[3])
    next_states_arr = np.asarray(next_states)
    next_states_arr = next_states_arr.reshape(32,32)
    #next_states_arr = next_states_arr.reshape(8, 32)
    return next_states_arr



# data_in = pd.read_csv("./dataset/real_data_trace.csv")
# data_in = data_in.drop("index",axis=1)
#设定随机种子
np.random.seed(40)

data_in = pd.read_csv("./dataset/perfectly_correlated.csv")

#设定DQN超参数

TIME_SLOTS = 5000
NUM_CHANNELS = 16               # Number of Channels
memory_size = 1000              # Experience Memory Size
batch_size = 32                 # Batch size for loss calculations (M)

step_size = 16
eps = 0.1                       # Exploration Probability
action_size = 16                # Action set size
state_size = 2                 # State Size (a_t-1, o_t-1 ,......, a_t-M,o_t-M)
learning_rate = 1e-2            # Learning rate
#gamma = 0.9                     # Discount Factor
hidden_size = 128                # Hidden Size (Put 200 for perfectly correlated)
pretrain_length = 16            # Pretrain Set to be known
n_episodes = 20                # Number of episodes (equivalent to epochs)

decay_epsilon_STEPS = 2000       #降低探索概率次数
UPDATE_PERIOD = 20  # update target network parameters目标网络随训练步数更新周期
#Lay_num_list = [50,50] #隐藏层节点设置
show_interval = 500  # To see loss trend iteration-wise (Put this 1 to see full trend)

# 超参数A_C
OUTPUT_GRAPH = False
MAX_EPISODE = 1
DISPLAY_REWARD_THRESHOLD = 200  # 刷新阈值
MAX_STEPS = 500 # 最大迭代次数
RENDER = False  # 渲染开关
GAMMA = 0.9  # 衰变值
LR_A = 0.0001  # Actor学习率
LR_C = 0.001  # Critic学习率


#DQN主程序
'''
if __name__ == "__main__":
    # 清空计算图
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7    #固定比例占用显存
    #config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    with tf.Session(config=config) as sess:
        DQN = DeepQNetwork(learning_rate=learning_rate, state_size=state_size,action_size=NUM_CHANNELS,
                            step_size=step_size, hidden_size=hidden_size, sess=sess)
        exp_memory = ExpMemory(in_size=memory_size)
        history_input = deque(maxlen=state_size*step_size)            #Input as states

        # Initialise the state of 16 actions and observations with random initialisation

        for i in range(pretrain_length):
            action = np.random.choice(action_size)
            obs = data_in["channel"+str(action)][i]
            history_input.append(action)
            history_input.append(obs)

        #prob_explore = 0.1  # Exploration Probability

        # loss_0 = []
        # avg_loss = []
        # reward_normalised = []

        #开始训练
        update_iter = 0
        loss_list = []
        reward_normalised_DRQN = []
        reward_normalised_r = []

        for episode in range(n_episodes):
            total_rewards = 0
            #loss_init = 0

            print("-------------Episode " + str(episode) + "-----------")
            for time in range(len(data_in) - pretrain_length):
            #for time in range(TIME_SLOTS):
                prob_sample = np.random.rand()
                state_in = np.array(history_input)  # Start State
                #print(np.shape(state_in))

                action = DQN.chose_action_train(state_in.reshape([-1,step_size,state_size]))  # 通过网络选择对应动作

                obs = data_in["channel" + str(int(action))][time + pretrain_length]  # Observe
                #print(int(action),obs)
                next_state = state_gen(state_in, action, obs)  # Go to next state
                reward = obs
                total_rewards += reward  # Total Reward
                exp_memory.add((state_in, action, reward, next_state))  # Add in exp memory
                state_in = next_state
                history_input = next_state

                if (time > state_size*step_size or episode != 0):  # If sufficient minibatch is available
                    batch = exp_memory.sample(batch_size)  # Sample without replacement
                    batch_states = get_states(batch)  # Get state,action,reward and next state from memory
                    batch_actions = get_actions(batch)
                    batch_rewards = get_rewards(batch)
                    batch_next_states = get_next_states(batch)

                    batch_states = np.reshape(batch_states,[-1,step_size,state_size])  # Get state,action,reward and next state from memory
                    # batch_actions = get_actions(batch)
                    # batch_rewards = get_rewards(batch)
                    batch_next_states = np.reshape(batch_next_states,[-1,step_size,state_size])

                    # print(np.shape(batch_next_states))
                    # print(np.shape(batch_states))
                    #print(np.shape(batch_rewards))
                    # print(np.shape(batch_actions))
                    summery, loss = DQN.train(state=batch_states,  # 进行训练
                                        reward=batch_rewards,
                                        action=batch_actions,
                                        state_next=batch_next_states
                                        )
                    #loss_init += loss
                    update_iter += 1
                    #loss_list.append(loss)
                    #DQN.write.add_summary(summery, update_iter)
                    # if (episode == 0):
                    #     loss_0.append(loss)

                    if (time % show_interval == 0):
                        print("Loss  at (t=" + str(time) + ") = " + str(loss))
                        print(int(action), obs)
                    if update_iter % UPDATE_PERIOD == 0:
                        DQN.update_prmt()  # 更新目标Q网络
                        #print("更新网络")

                    if time % decay_epsilon_STEPS == 0:
                        DQN.decay_epsilon()  # 随训练进行减小探索力度
            print("Total Reward: ")
            print(total_rewards / len(data_in))

                # # Plot Display of Loss in episode 0
                # if (time == len(data_in) - pretrain_length - 1 and episode == 0):
                # #if (time == TIME_SLOTS and episode == 0):
                #     plt.plot(loss_0)
                #     plt.xlabel("Iteration")
                #     plt.ylabel("Q Loss")
                #     plt.title('Iteration vs Loss (Episode 0)')
                #     plt.show()

        print("-------------DQN测试 -----------")
        r_t_DRQN = []
        a_t_DRQN = []
        total_rewards = 0
        for time in range(len(data_in) - pretrain_length):
            # for time in range(TIME_SLOTS):
            prob_sample = np.random.rand()
            state_in = np.array(history_input)  # Start State
            # print(np.shape(state_in))

            action = DQN.chose_action_test(state_in.reshape([-1, step_size, state_size]))  # 通过网络选择对应动作

            obs = data_in["channel" + str(int(action))][time + pretrain_length]  # Observe
            next_state = state_gen(state_in, action, obs)  # Go to next state
            reward = obs
            r_t_DRQN.append(reward)
            a_t_DRQN.append(int(action))
            #print(action,obs)
            total_rewards += reward  # Total Reward
            reward_normalised_DRQN.append(total_rewards)
            #exp_memory.add((state_in, action, reward, next_state))  # Add in exp memory
            state_in = next_state
            history_input = next_state
        print("DQN_Total Reward: ")
        print(total_rewards / len(data_in))

        print("-------------随机对比 -----------")
        total_rewards = 0
        r_r = []
        a_r = []
        for time in range(len(data_in) - pretrain_length):
            # for time in range(TIME_SLOTS):
            prob_sample = np.random.rand()
            state_in = np.array(history_input)  # Start State
            # print(np.shape(state_in))

            action = np.random.randint(0, action_size)  # 通过网络选择对应动作
            # print(action)
            obs = data_in["channel" + str(int(action))][time + pretrain_length]  # Observe
            next_state = state_gen(state_in, action, obs)  # Go to next state
            reward = obs
            r_r.append(reward)
            a_r.append(action)
            total_rewards += reward  # Total Reward
            reward_normalised_r.append(total_rewards)
            #exp_memory.add((state_in, action, reward, next_state))  # Add in exp memory
            state_in = next_state
            history_input = next_state
        print("RANDOM_Total Reward: ")
        print(total_rewards / len(data_in))

            # #Average loss
            # print("Average Loss: ")
            # print(loss_init/(len(data_in)))
            # #Average reward observed in full iterations
            # print("Total Reward: ")
            # print(total_rewards/len(data_in))
            # avg_loss.append(loss_init/(len(data_in)))
            # reward_normalised.append(total_rewards/len(data_in))

        # See reward and loss trend episode wise

        # print(reward_all_t, TIME_SLOTS * 3)
        # print(reward_all_r, TIME_SLOTS * 3)
        # array1 = np.array(r_r)
        # r_r_data = pd.DataFrame(array1)
        # r_r_data.to_csv("./dataset/random_channl_reward.csv")
        #
        # array3 = np.array(a_r)
        # a_r_data = pd.DataFrame(array3)
        # a_r_data.to_csv("./dataset/random_channl_action.csv")
        #
        # array2 = np.array(r_t)
        # r_t_data = pd.DataFrame(array2)
        # r_t_data.to_csv("./dataset/DRQN_channl_reward.csv")
        #
        # array4 = np.array(a_t)
        # a_t_data = pd.DataFrame(array4)
        # a_t_data.to_csv("./dataset/DRQN_channl_action.csv")
        #
        # array5 = np.array(loss_list)
        # loss_data = pd.DataFrame(array5)
        # loss_data.to_csv("./dataset/DRQN_channl_choose_loss.csv")

        # plt.figure(1)
        # # plt.subplot(121)
        # # plt.plot(np.arange(len(loss_list)), loss_list, "r-")
        # # plt.xlabel('Time Slots')
        # # plt.ylabel('total loss')
        # # plt.title('total loss given per time_step')
        # # plt.subplot(122)
        # plt.plot(np.arange(len(reward_normalised_t)), reward_normalised_t, "b-")
        # plt.plot(np.arange(len(reward_normalised_r)), reward_normalised_r, "r:")
        # plt.xlabel('Time Slots')
        # plt.ylabel('total rewards')
        # plt.legend(['DRQN', 'Random'])
        # plt.title('total rewards given per time_step')
        # plt.show()

'''


#A_C主程序
if __name__ == "__main__":
    # 清空计算图
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7    #固定比例占用显存
    #config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    with tf.Session(config=config) as sess:
        actor = Actor(sess, n_features=state_size*step_size, n_actions=action_size, lr=LR_A)  # 初始化Actor
        critic = Critic(sess, n_features=state_size*step_size, lr=LR_C)  # 初始化Critic
        sess.run(tf.global_variables_initializer())  # 初始化参数
        #exp_memory = ExpMemory(in_size=memory_size)
        history_input = deque(maxlen=state_size*step_size)            #Input as states

        # Initialise the state of 16 actions and observations with random initialisation

        for i in range(pretrain_length):
            action = np.random.choice(action_size)
            obs = data_in["channel"+str(action)][i]
            history_input.append(action)
            history_input.append(obs)

        #prob_explore = 0.1  # Exploration Probability

        # loss_0 = []
        # avg_loss = []
        # reward_normalised = []

        #开始训练
        update_iter = 0
        loss_list = []
        reward_normalised_AC = []
        #reward_normalised_r = []

        for episode in range(n_episodes):
            total_rewards = 0
            #loss_init = 0
            print("-------------Episode " + str(episode) + "-----------")
            for time in range(len(data_in) - pretrain_length):
            #for time in range(TIME_SLOTS):
                prob_sample = np.random.rand()
                state_in = np.array(history_input)  # Start State
                #print(np.shape(state_in))

                action = actor.choose_action(state_in.reshape([step_size*state_size]))  # 通过网络选择对应动作

                obs = data_in["channel" + str(int(action))][time + pretrain_length]  # Observe
                #print(int(action),obs)
                next_state = state_gen(state_in, action, obs)  # Go to next state
                #reward = obs
                reward = obs*2-1
                total_rewards += reward  # Total Reward
                td_error = critic.learn(state_in, reward, next_state)  # Critic 学习
                actor.learn(state_in, action, td_error)  # Actor 学习
                #exp_memory.add((state_in, action, reward, next_state))  # Add in exp memory
                state_in = next_state
                history_input = next_state

                # if (time > state_size*step_size or episode != 0):  # If sufficient minibatch is available
                #     batch = exp_memory.sample(batch_size)  # Sample without replacement
                #     batch_states = get_states(batch)  # Get state,action,reward and next state from memory
                #     batch_actions = get_actions(batch)
                #     batch_rewards = get_rewards(batch)
                #     batch_next_states = get_next_states(batch)
                #
                #     batch_states = np.reshape(batch_states,[-1,step_size,state_size])  # Get state,action,reward and next state from memory
                #     # batch_actions = get_actions(batch)
                #     # batch_rewards = get_rewards(batch)
                #     batch_next_states = np.reshape(batch_next_states,[-1,step_size,state_size])
                #
                #     # print(np.shape(batch_next_states))
                #     # print(np.shape(batch_states))
                #     #print(np.shape(batch_rewards))
                #     # print(np.shape(batch_actions))

                update_iter += 1
                loss_list.append(td_error**2)
                    #DQN.write.add_summary(summery, update_iter)
                    # if (episode == 0):
                    #     loss_0.append(loss)

                # if (time <= 200):
                #     print("Loss  at (t=" + str(time) + ") = " + str(td_error ** 2))
                #     print(int(action), obs)

                if update_iter % decay_epsilon_STEPS == 0:
                    actor.decay_epsilon()  # 随训练进行减小探索力度

                if (time % show_interval == 0):
                    print("Loss  at (t=" + str(time) + ") = " + str(td_error**2))
                    print(int(action), obs)
                    # if update_iter % UPDATE_PERIOD == 0:
                    #     DQN.update_prmt()  # 更新目标Q网络
                    #     #print("更新网络")

                    # if time % decay_epsilon_STEPS == 0:
                    #     DQN.decay_epsilon()  # 随训练进行减小探索力度
            print("A_C_Total Reward: ")
            print(total_rewards / len(data_in))

                # # Plot Display of Loss in episode 0
                # if (time == len(data_in) - pretrain_length - 1 and episode == 0):
                # #if (time == TIME_SLOTS and episode == 0):
                #     plt.plot(loss_0)
                #     plt.xlabel("Iteration")
                #     plt.ylabel("Q Loss")
                #     plt.title('Iteration vs Loss (Episode 0)')
                #     plt.show()

        print("-------------A_C测试 -----------")
        r_t_AC = []
        a_t_AC = []
        total_rewards = 0
        for time in range(len(data_in) - pretrain_length):
            # for time in range(TIME_SLOTS):
            prob_sample = np.random.rand()
            state_in = np.array(history_input)  # Start State
            # print(np.shape(state_in))

            action = actor.choose_action(state_in.reshape([ step_size * state_size]))  # 通过网络选择对应动作

            obs = data_in["channel" + str(int(action))][time + pretrain_length]  # Observe
            # print(int(action),obs)
            next_state = state_gen(state_in, action, obs)  # Go to next state
            reward = 2*obs-1
            total_rewards += reward  # Total Reward
            td_error = critic.learn(state_in, reward, next_state)  # Critic 学习
            actor.learn(state_in, action, td_error)  # Actor 学习
            # exp_memory.add((state_in, action, reward, next_state))  # Add in exp memory
            state_in = next_state
            history_input = next_state
            r_t_AC.append(reward)
            a_t_AC.append(int(action))
            #print(action,obs)
            reward_normalised_AC.append(total_rewards)
            #exp_memory.add((state_in, action, reward, next_state))  # Add in exp memory
        print("AC_Total Reward: ")
        print(total_rewards / len(data_in))

        print("-------------随机对比 -----------")
        total_rewards = 0
        r_r = []
        a_r = []
        reward_normalised_r = []
        for time in range(len(data_in) - pretrain_length):
            # for time in range(TIME_SLOTS):
            prob_sample = np.random.rand()
            state_in = np.array(history_input)  # Start State
            # print(np.shape(state_in))

            action = np.random.randint(0, action_size)  # 通过网络选择对应动作
            # print(action)
            obs = data_in["channel" + str(int(action))][time + pretrain_length]  # Observe
            next_state = state_gen(state_in, action, obs)  # Go to next state
            reward = 2*obs-1
            r_r.append(reward)
            a_r.append(action)
            total_rewards += reward  # Total Reward
            reward_normalised_r.append(total_rewards)
            # exp_memory.add((state_in, action, reward, next_state))  # Add in exp memory
            state_in = next_state
            history_input = next_state
        print("RANDOM_Total Reward: ")
        print(total_rewards / len(data_in))

        print("-------------对比结果 -----------")
        # print(a_t_DRQN)
        print(a_t_AC)


        # plt.figure(1)
        # # plt.subplot(121)
        # # plt.plot(np.arange(len(loss_list)), loss_list, "r-")
        # # plt.xlabel('Time Slots')
        # # plt.ylabel('total loss')
        # # plt.title('total loss given per time_step')
        # # plt.subplot(122)
        #
        # plt.plot(np.arange(len(reward_normalised_r)), reward_normalised_DRQN, "y-")
        # plt.plot(np.arange(len(reward_normalised_r)), reward_normalised_AC, "b-")
        # plt.plot(np.arange(len(reward_normalised_r)), reward_normalised_r, "r:")
        # plt.xlabel('Time Slots')
        # plt.ylabel('total rewards')
        # plt.legend(['DRQN','A_C', 'Random'])
        # plt.title('total rewards given per time_step')
        # plt.show()
        loss_list=np.array(loss_list)
        loss_list=loss_list[:,0,0]

        plt.figure(1)
        plt.plot(np.arange(len(loss_list)), loss_list, "r-")
        plt.xlabel('Time Slots')
        plt.ylabel('total loss')
        plt.title('total loss given per time_step')
        plt.figure(2)
        plt.plot(np.arange(len(reward_normalised_AC)), reward_normalised_AC, "b-")
        plt.plot(np.arange(len(reward_normalised_r)), reward_normalised_r, "r:")
        plt.xlabel('Time Slots')
        plt.ylabel('total rewards')
        plt.legend([ 'A_C', 'Random'])
        plt.title('total rewards given per time_step')
        plt.show()