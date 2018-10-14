import numpy as np
import FlockingEnv as fe

from ddpg_agent import DDPGAgent, MINI_BATCH_SIZE
from ou_noise import OUNoise


# parameters
episodes_num = 20000
is_movie_on = False
size1 = 50
dim = 2
steps_limit = 1000

def main():
    # Instanciate specified environment.
    env = fe.FlockingEnv(size1,dynamic="first")

    # Get environment specs
    num_states = (size1+1) * dim * 2
    num_actions = size1 *dim

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
    noise = OUNoise(1)

    for i in range(episodes_num):
        print("--------Episode %d--------" % i)
        reward_per_episode = 0
        observation = env.reset_mul()

        for j in range(steps_limit):
            if is_movie_on: env.render()

            # Select action off-policy
            state = observation
            action = np.zeros((size1,dim),dtype = np.float32)
            # get individual ob states here
            for k in range(0, size1):
                ac = agent.feed_forward_actor(np.reshape(state[k], [1, num_states]))
                # print(noise.generate())
                if i%2 == 0:
                    action[k][0]=ac[0][0]  + noise.generate()
                    action[k][1]=ac[0][1]  + noise.generate()
                else:
                    action[k][0] = ac[0][0] + noise.generate()
                    action[k][1] = ac[0][1] + noise.generate()

            # Throw action to environment
            observation, reward, done, info = env.step_mul(action)

            for k in range(0,size1):
                agent.add_experience(np.reshape(state[k], [num_states]), action[k],
                                                np.reshape(observation[k], [ num_states]), reward[k], done)

            # Train actor/critic network
            if len(agent.replay_buffer) > MINI_BATCH_SIZE: agent.train()

            reward_per_episode = reward.sum()

            if j % 100 == 0:
                print(j,"step finished. reward=",reward_per_episode,"info=",info)
                # print("action=",action,"observation=",observation)
            if (done or j == steps_limit -1):
                print("Steps count: %d" % j)
                print("Total reward: %d" % reward_per_episode)

                env.render()
                #noise.reset()

                with open("reward_log.csv", "a") as f:
                    f.write("%d,%f\n" % (i, reward_per_episode))

                break

if __name__ == "__main__":
    main()

