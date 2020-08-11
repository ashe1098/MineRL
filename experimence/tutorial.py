# -*- coding: utf-8 -*-
import gym
import minerl

import tqdm
import numpy as np
from sklearn.cluster import KMeans

import logging
logging.basicConfig(level=logging.DEBUG)  # ログを見れる
# import coloredlogs
# coloredlogs.install(logging.INFO)

def random():
    # env = gym.make('MineRLNavigateDense-v0')
    env = gym.make('MineRLTreechopVectorObf-v0')

    # set the environment to allow interactive connections on port 6666
    # and slow the tick speed to 6666.
    env.make_interactive(port=6666, realtime=True)

    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # ランダムに動く
        obs, reward, done, _ = env.step(action)
        env.render()

def target():
    env = gym.make('MineRLNavigateDense-v0')
    # env = gym.wrappers.Monitor(env, "/Users/ashe/workspace/Creation/Capture")  # 記録用

    env.make_interactive(port=6666, realtime=True)
    obs  = env.reset()
    done = False
    net_reward = 0
    while not done:  # 目的地に向かって動く
        action = env.action_space.noop()

        action['camera'] = [0, 0.03*obs["compassAngle"]]
        action['back'] = 0
        action['forward'] = 1
        action['jump'] = 0
        action['attack'] = 1

        obs, reward, done, info = env.step(action)
        env.render()

        net_reward += reward
        print("Total reward: ", net_reward)


def kmeans():

    # dat = minerl.data.make('MineRLTreechopVectorObf-v0')  # $ MINERL_DATA_ROOT="/Users/ashe/workspace/MineRL/data"
    dat = minerl.data.make('MineRLTreechopVectorObf-v0', data_dir="/Users/ashe/workspace/MineRL/data")
    # dat = minerl.data.make('MineRLNavigateVectorObf-v0', data_dir="/Users/ashe/workspace/MineRL/data")
    # dat = minerl.data.make('MineRLObtainDiamondVectorObf-v0', data_dir="/Users/ashe/workspace/MineRL/data")

    act_vectors = []
    NUM_CLUSTERS = 4

    # Load the dataset storing 1000 batches of actions
    for _, act, _, _, _ in tqdm.tqdm(dat.batch_iter(16, 32, 2, preload_buffer_size=20)):
        act_vectors.append(act['vector'])
        if len(act_vectors) > 1000:
            break

    # Reshape these the action batches
    acts = np.concatenate(act_vectors).reshape(-1, 64)
    kmeans_acts = acts[:100000]  # 使う情報は100000個のみ

    # Use sklearn to cluster the demonstrated actions
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0).fit(kmeans_acts)
    print ('Distortion: %.2f'% kmeans.inertia_)  # 小さい方がいい（としか言えない）．0が最適


    # # 図やグラフを図示するためのライブラリをインポートする。
    # # import matplotlib
    # # matplotlib.use('TkAgg')  # 古いバージョンだとbackend: macosxに対応していないので，指定し直す
    # import matplotlib.pyplot as plt
    # from pandas import plotting # 高度なプロットを行うツールのインポート
    # #import sklearn #機械学習のライブラリ
    # from sklearn.decomposition import PCA #主成分分析器
    # labels = kmeans.labels_
    # #主成分分析の実行
    # # 64次元特徴を2次元にする
    # pca = PCA(n_components=2)
    # pca.fit(kmeans_acts)
    # pca_data = pca.fit_transform(kmeans_acts)

    # from colorsys import hls_to_rgb
    # def get_distinct_colors(n):
    #     colors = []
    #     for i in np.arange(0., 360., 360. / n):
    #         h = i / 360.
    #         l = (50 + np.random.rand() * 10) / 100.
    #         s = (90 + np.random.rand() * 10) / 100.
    #         colors.append(hls_to_rgb(h, l, s))
    #     return colors
    
    # # それぞれに与える色を決める。
    # color_codes = get_distinct_colors(NUM_CLUSTERS)
    # # サンプル毎に色を与える。
    # colors = [color_codes[x] for x in labels]
    # # クラスタリング結果のプロット
    # plt.figure()
    # plt.scatter(pca_data[:,0], pca_data[:,1], c=colors)
    # # for i in range(pca_data.shape[0]):  # 上と同じ
    # #     plt.scatter(pca_data[i,0], pca_data[i,1], c=color_codes[int(labels[i])])
    # plt.title("Principal Component Analysis")
    # plt.xlabel("The first principal component score")
    # plt.ylabel("The second principal component score")
    # plt.savefig("dpi_scatter.png", format="png", dpi=300)
    # plt.show()


    i, net_reward, done, env = 0, 0, False, gym.make('MineRLTreechopVectorObf-v0')
    env.make_interactive(port=6666, realtime=True)
    obs = env.reset()

    while not done:
        # Let's use a frame skip of 4 (could you do better than a hard-coded frame skip?)
        if i % 4 == 0:
            action = {  # 変えてもダメ
                # 'vector': kmeans.cluster_centers_[np.random.choice(NUM_CLUSTERS)] # 4フレームごとにクラスタリングした行動のどれかを選択する
                'vector': kmeans.cluster_centers_[(int)((i/4) % NUM_CLUSTERS)]  # 4フレームごとにクラスタリングした行動を順にとる
            }

            obs, reward, done, info = env.step(action)

            env.render()

            if reward > 0:
                print("+{} reward!".format(reward))
            net_reward += reward
        i += 1

    print("Total reward: ", net_reward)


def no_op_test():
    env = gym.make("MineRLTreechopVectorObf-v0")
    while True:
        obs = env.reset()
        done= False
        while not done:
            x = env.env_spec.wrap_action(env.env_spec.env_to_wrap.env_to_wrap.action_space.no_op())
            print(x)
            a,r,done,i = env.step(x) # Also happens if you try to feed in {'vector': np.random.random((64,))}
            env.render()


def simple_env_test():
    """
    Tests running a simple environment.
    """
    NUM_EPISODES=1
    env = gym.make('MineRLNavigateDense-v0')
    
    actions = [env.action_space.sample() for _ in range(2000)]
    xposes = []
    env.seed(25)
    for _ in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        netr = 0
        while not done:
            random_act = env.action_space.noop()
            
            random_act['camera'] = [0, 0.1*obs["compassAngle"]]  # compassAngleはNavigationのみ
            random_act['back'] = 0
            random_act['forward'] = 1
            random_act['jump'] = 1
            random_act['attack'] = 1
            obs, reward, done, info = env.step(
                random_act)
            netr += reward
            print(reward, netr)
            env.render()
    print("Demo complete.")


def simple_treechop_test():  # env.resetでエラー
    """
    Tests running a simple environment.
    """
    NUM_EPISODES=6
    env = gym.make('MineRLTreechop-v0')  # データセットが悪さしてそう
    
    actions = [env.action_space.sample() for _ in range(2000)]
    xposes = []
    for i in range(NUM_EPISODES):
        env.seed(i)
        obs = env.reset()  # 実行できてもTypeError: a bytes-like object is required, not 'NoneType'
        done = False
        netr = 0
        for _ in range(i):
            random_act = env.action_space.noop()
            # random_act['camera'] = [0, 0.1]
            # random_act['back'] = 0
            # random_act['forward'] = 1
            # random_act['jump'] = 1
            obs, reward, done, info = env.step(
                random_act)
            netr += reward
            print(reward, netr)
            env.render()
    print("Demo complete.")


def seed_test():
    """
    Tests running a simple environment.
    """
    NUM_EPISODES=10
    env = gym.make('MineRLNavigateDense-v0')  # MineRLTreechop-v0で動く？
    
    actions = [env.action_space.sample() for _ in range(2000)]
    xposes = []
    reward_list = []
    for _ in range(NUM_EPISODES):
        env.seed(22)
        obs = env.reset()  # 木にぶつかってTypeError: a bytes-like object is required, not 'NoneType'  ループが悪そう！
        done = False
        netr = 0
        rewards = []
        while not done and  len(rewards) < 50:
            random_act = env.action_space.noop()
            # if(len(rewards) > 50):
            
            random_act['camera'] = [0, 0.1]
            random_act['back'] = 0
            random_act['forward'] = 1
            random_act['jump'] = 1
            random_act['attack'] = 1
            # print(random_act)
            obs, reward, done, info = env.step(
                random_act)
            env.render()
            rewards.append(reward)
            netr += reward
            # print(reward, netr)
        reward_list.append(rewards)
    import matplotlib.pyplot as plt
    for t in range(NUM_EPISODES):
        plt.plot(np.cumsum(reward_list[t]))
    # plt.plot(np.cumsum(reward_list[1]))
    plt.show()
    # from IPython import embed; embed()
    input()
    print("Demo complete.")


def data_check():
    # Sample some data from the dataset!
    data = minerl.data.make("MineRLTreechopVectorObf-v0", data_dir="/Users/ashe/workspace/MineRL/data")

    # Iterate through batches of data
    counter = 0
    # batch_size: データの個数（=行数）, seq_len: 各データのステップ（フレーム）数（=列数）, rew=reward（報酬）
    for obs,  act, rew,  next_obs, done in data.batch_iter(batch_size=2, seq_len=5, num_epochs=1):
        # Do something
        correct_len = len(rew)
        print("Obs shape:", obs)  # pov, vector
        print("Act shape:", act)  # vector
        # print("Rew shape:", rew)
        # print("Done shape:", done)
        # print(counter + 1)
        # counter += 1


def kmeans_check():
    dat = minerl.data.make("MineRLTreechopVectorObf-v0", data_dir="/Users/ashe/workspace/MineRL/data")
    act_vectors = []
    NUM_CLUSTERS = 30

    # Load the dataset storing 1000 batches of actions
    for _, act, _, _, _ in tqdm.tqdm(dat.batch_iter(3, 32, 2)):
        act_vectors.append(act['vector'])
        print(len(act_vectors))
        if len(act_vectors) > 100:
            break

    # print(act_vectors)  # 行列のリスト
    print(np.concatenate(act_vectors).shape)  # 行列に変換
    # Reshape these the action batches
    acts = np.concatenate(act_vectors).reshape(-1, 64)
    print(acts.shape) # 時系列情報は消える


def render_on_ipython():
    import matplotlib.pyplot as plt
    from IPython import display

    env = gym.make('MineRLNavigateDenseVectorObf-v0')

    obs = env.reset()
    done = False
    logging.basicConfig(level=logging.CRITICAL)
    while not done:
        action = env.action_space.sample()  # ランダムに動く
        obs, reward, done, _ = env.step(action)
        plt.imshow(env.render())  # loggingはオフにしておくこと！
        display.display(plt.gcf())
        display.clear_output(wait=True)
    logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    random()
