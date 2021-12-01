
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


file_path = '/Users/dandanchen/Desktop/703data/'
# sub_file = 'test5/'
# sub_file = 'her_test2/'
sub_file_list = ['']
for sub_file in sub_file_list:
    file_name = 'trainCP'
    data = pd.read_csv(file_path+sub_file+file_name, sep=",", header=None)
    data.columns = ["succ", "mean_r", "std_r", "mean_q", "std_q"]

    succ = data.succ.values
    mean_r = data.mean_r.values
    std_r = data.std_r.values
    mean_q = data.mean_q.values
    std_q = data.std_q.values
    print(len(std_q))
    print((mean_q))
    print(succ.shape[0])

    plt.errorbar(np.linspace(1, succ.shape[0], succ.shape[0]), mean_q, yerr=std_q, ecolor='r', color='b')
    # plt.ylim([0,300])
    # plt.xlim([0,370])
    plt.axhline(y=0)
    plt.xlabel('iterations/100')
    plt.ylabel('average predicted Q from critic')
    name = 'average_predicted_Q_DDPG'
    plt.title(name)
    plt.savefig(file_path + sub_file + name)
    plt.close()

    plt.errorbar(np.linspace(1, succ.shape[0], succ.shape[0]), mean_r, yerr=std_r, ecolor='r', color='b')
    # plt.ylim([0,300])
    # plt.xlim([0,370])
    plt.axhline(y = 0)
    plt.xlabel('iterations/100')
    plt.ylabel('average reward')
    name = 'average_reward_DDPG'
    plt.title(name)
    plt.savefig(file_path + sub_file + name)
    plt.close()
    plt.plot(np.linspace(1, succ.shape[0], succ.shape[0]), succ, 'm')
    plt.axhline(y = 0.95)
    plt.xlabel('iterations/100')
    plt.ylabel('succ rate')
    name = 'succ_rate_DDPG'
    plt.title(name)
    plt.savefig(file_path + sub_file + name)
    plt.close()