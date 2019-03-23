import gym

from DQN import *
env = gym.make('CartPole-v0')
RL = DeepQNetwork(2, 4,
                  learning_rate=0.001,
                  reward_decay=0.9,
                  replace_target_iter=200,
                  memory_size=2000,
                  output_graph=True
                  )

result = []
# checkpoint = tf.train.get_checkpoint_state('models')
# RL.saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
# RL.saver.restore(RL.sess, checkpoint.model_checkpoint_path)
for i_episode in range(180):
    observation = env.reset()
    for t in range(10000):
        env.render()
        action = RL.choose_action(observation)
        observation_, reward, done,info = env.step(action)
        RL.store_transition(observation, action, reward, observation_)
        observation = observation_
        if i_episode >= 100:
            RL.learn()
        if done:
            result.append(t)
            print(i_episode," Finished in {} timesteps".format(t+1))
            break
RL.save_model()
env.close()