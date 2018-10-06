import numpy as np
import gym
import FlockingEnv

from ddpg_agent import DDPGAgent, MINI_BATCH_SIZE
from ou_noise import OUNoise


# parameters
episodes_num = 20000
is_movie_on = True
size = 50
dim = 2
steps_limit = 1000

def main():
    # Instanciate specified environment.
    env = FlockingEnv(size)

    # Get environment specs
    num_states = size * dim * 2
    num_actions = size *dim

    # Print specs
    print("Number of states: %d" % num_states)
    print("Number of actions: %d" % num_actions)
    print("-----------------------------------------")

    # Instanciate reinforcement learning agent which contains Actor/Critic DNN.
    #agents =[]
    #for i in range(0,size):
    agent = DDPGAgent(ob_shape=num_states, ac_shape=dim)
    #    agents.append(agent)
    # Exploration noise generator which uses Ornstein-Uhlenbeck process.
    noise = OUNoise(dim)

    for i in range(episodes_num):
        print("--------Episode %d--------" % i)
        reward_per_episode = 0
        observation = env.reset_mul()

        for j in range(steps_limit):
            if is_movie_on: env.render()

            # Select action off-policy
            state = observation
            action = np.zeros((size,dim),dtype = np.float32)
            # get individual ob states here
            for i in range(0, size):
                ac = agent.feed_forward_actor(np.reshape(state[i], [1, num_states]))
                action[i]=ac + noise.generate()

            # Throw action to environment
            observation, reward, done, info = env.step_mul(action)

            for i in range(0,size):
                agent.add_experience(state[i], action[i], observation[i], reward[i], done)

            # Train actor/critic network
            if len(agent.replay_buffer) > MINI_BATCH_SIZE: agent.train()

            reward_per_episode += reward

            if (done or j == steps_limit -1):
                print("Steps count: %d" % j)
                print("Total reward: %d" % reward_per_episode)

                #noise.reset()

                with open("reward_log.csv", "a") as f:
                    f.write("%d,%f\n" % (i, reward_per_episode))

                break

if __name__ == "__main__":
    main()

