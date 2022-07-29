import scipy.io as sio
import numpy as np
import mat73
import sys
sys.path.append('/home/katrina/a/mankovic/FlagIRLS')
import center_algorithms as ca
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
from sklearn.manifold import MDS


n_its= 10
seed = 0
n_trials = 20

f_name = './minds_eye_lbg_'+str(n_trials)+'trials.png'

labels_raw = sio.loadmat('/data4/mankovic/CVPR2022/data/MindsEye/kmeans_action_labels.mat')['kmeans_action_labels']

labels_true = [l[0][0] for l in labels_raw['labels'][0][0]]
# labelidxs =labels_raw['labelidxs'][0][0][0]


raw_data = mat73.loadmat('/data4/mankovic/CVPR2022/data/MindsEye/kmeans_pts.mat')

X = [t[0] for t in raw_data['Data']['gr_pts']]



idx = []
for the_labels in ['run', 'pickup', 'bend','follow', 'ride-bike']:
# for the_labels in ['run', 'stand', 'pickup']:
# for the_labels in ['run', 'stand', 'walk-rifle']: #for winning sine median
    idx += list(np.where(np.array(labels_true) == the_labels)[0])

labels_true = [labels_true[i] for i in idx]
print(labels_true)
X = [X[i] for i in idx]


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
        sin_purity = ca.cluster_purity(X, centers_sin, labels_true, 'sine')
        print('cos start')
        centers_cos, error_cos, dist_cos = ca.lbg_subspace(X, .0001, n_centers = n, opt_type = 'cosine', n_its = 10, seed = trial, similarity = True)
        cos_purity = ca.cluster_purity(X, centers_cos, labels_true, 'cosine', similarity = True)
        print('l2 start')
        centers_l2, error_l2, dist_l2 = ca.lbg_subspace(X, .0001, n_centers = n, opt_type = 'l2_med', n_its = 10, seed = trial)
        l2_purity = ca.cluster_purity(X, centers_l2, labels_true, 'l2_med')
        print('flg start')
        centers_flg, error_flg, dist_flg = ca.lbg_subspace(X, .0001, n_centers = n, opt_type = 'sinesq', seed = trial)
        flg_purity = ca.cluster_purity(X, centers_flg, labels_true, 'sinesq')


        Purities = Purities.append({'Algorithm': 'Flag Median', 
                                'Codebook Size': n,
                                'Cluster Purity': sin_purity},
                                ignore_index = True)
        Purities = Purities.append({'Algorithm': 'L2 Median', 
                                'Codebook Size': n,
                                'Cluster Purity': l2_purity},
                                ignore_index = True)
        Purities = Purities.append({'Algorithm': 'Flag Mean', 
                                'Codebook Size': n,
                                'Cluster Purity': flg_purity},
                                ignore_index = True)
        Purities = Purities.append({'Algorithm': 'Max Cor Flag', 
                                'Codebook Size': n,
                                'Cluster Purity': cos_purity},
                                ignore_index = True)
    print(Purities)
Purities.to_csv('minds_eye_lbg_results.csv')
        


sns.boxplot(x = 'Codebook Size', y = 'Cluster Purity', hue = 'Algorithm', data = Purities)
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.savefig(f_name, bbox_inches='tight')

# Purities.to_csv('LBG_results_20trials.csv')