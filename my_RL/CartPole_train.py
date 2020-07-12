#from mMaze import Maze
from DQN import DQN
import gym
if __name__ =="__main__":
    env = gym.make('CartPole-v0') #建立环境
    env= env.unwrapped

#以下可以显示这个环境的state 和 action

    print(env.action_space)
    print(env.observation_space.shape[0])
    print(env.observation_space.high)
    print(env.observation_space.low)

#初始化DQN的模型
    RL= DQN(n_actions = env.action_space.n,
            n_features = env.observation_space.shape[0],
            learning_rate = 0.01,
            e_greedy = 0.9,
            replace_target_iter = 100,
            memory_size = 2000,
            e_greedy_increment = 0.001,
            use_e_greedy_increment = 1000)

    steps = 0
    for episode in range(300): #训练300个回合，这里环境模型，结束回合的标志是 倾斜程度和 X 的移动限度，你可以很容易从训练效果中看出来，当然了，也可以去看gym的底层代码，还是比较清晰的。

        observation = env.reset()
        ep_r = 0

        while True: #训练没有结束的时候循环
            env.render()#刷新环境
            action = RL.choose_action(observation)#根据状态选择行为
            observation_next,reward,done,info = env.step(action)#环境模型 采用行为，获得下个状态，和潜在的奖励

            x,x_dot,theta,theat_dot = observation_next#这里拆分了 状态值 ，里面有四个参数
#这里用了，x 和theta的限度值 来判断奖励的幅度，当然也可以gym自带的 ，但是这个效率据说比较高
            reward1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
            reward2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
            reward = reward1+reward2 #这里将奖励综合

            RL.store_transition(observation,action,reward,observation_next)  #先存储到记忆库
            ep_r+= reward #这里只是为了观察奖励值是否依据实际情况变化，来方便判断模型的正确性
            if steps>1000: #这里一开始先不学习，先积累奖励
                RL.learn()
            if done :  #这里判断的是回合结束，显示奖励积累值，你可以看到每回合奖励的变化，来判定这样一连串行为的结果好不好
                print('episode :',episode,
                      'ep_r:',round(ep_r,2),
                      "RL's epsilon",round(RL.epsilon,3))
                break

            observation = observation_next #跟新状态
            steps+=1
    RL.plot_cost()
