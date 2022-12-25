import numpy as np
from matplotlib import pyplot as plt
import center_algorithms as ca
from os import listdir
import pandas
import seaborn as sns




n_its= 10
seed = 0
n_trials = 10


base_path = '/data4/mankovic/CVPR2022/data/action_youtube_gr/'
X = []
labels_true = []
count = 0
for label in listdir(base_path):
    if count < 5:
        current_dir = base_path+label+'/'
        for f in listdir(current_dir):
            X.append(np.load(current_dir+f))
            labels_true.append(label)
    count += 1



f_name = './youtube_lbg_'+str(n_trials)+'trials.png'



Purities = pandas.DataFrame(columns = ['Algorithm','Codebook Size','Cluster Purity'])

for n in range(4, 24, 4):
    sin_purities = []
    cos_purities = []
    flg_purities = []
    for trial in range(n_trials):
        print('cluster '+str(n)+' trial '+str(trial))
        print('.')
        print('.')
        print('.')
        print('sin start')
        centers_sin, error_sin, dist_sin = ca.lbg_subspace(X, .0001, n_centers = n, opt_type = 'sine', n_its = 10, seed = trial)
        sin_purity = ca.cluster_purity(X, centers_sin, labels_true)
        # print('cos start')
        # centers_cos, error_cos, dist_cos = ca.lbg_subspace(X, .0001, n_centers = n, opt_type = 'cosine', n_its = 10, seed = trial, similarity = False)
        # cos_purity = ca.cluster_purity(X, centers_cos, labels_true, similarity = False)
        print('flg start')
        centers_flg, error_flg, dist_flg = ca.lbg_subspace(X, .0001, n_centers = n, opt_type = 'sinesq', seed = trial)
        flg_purity = ca.cluster_purity(X, centers_flg, labels_true)


        Purities = Purities.append({'Algorithm': 'Flag Median', 
                                'Codebook Size': n,
                                'Cluster Purity': sin_purity},
                                ignore_index = True)
        
        # Purities = Purities.append({'Algorithm': 'Max Cor Flag', 
        #                         'Codebook Size': n,
        #                         'Cluster Purity': cos_purity},
        #                         ignore_index = True)

        Purities = Purities.append({'Algorithm': 'Flag Mean', 
                                'Codebook Size': n,
                                'Cluster Purity': flg_purity},
                                ignore_index = True)
    # print(Purities)
    # Purities.to_csv('youtube_LBG_results_20trials'+str(n)+'.csv')
        
Purities.to_csv('./youtube_lbg_'+str(n_trials)+'trials.csv')

sns.boxplot(x='Codebook Size', y='Cluster Purity', hue='Algorithm', data = Purities)
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.savefig(f_name, bbox_inches='tight')


