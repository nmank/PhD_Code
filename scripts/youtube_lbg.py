import numpy as np
from matplotlib import pyplot as plt
import center_algorithms as ca
from os import listdir
import pandas
import seaborn as sns

def distance_matrix(X, C, opt_type = 'sine'):
    n = len(X)
    m = len(C)
    Distances = np.zeros((m,n))

    # if opt_type == 'cosine':
    #     opt_type = 'sinesq'

    for i in range(m):
        for j in range(n):
            Distances[i,j] = ca.calc_error_1_2([C[i]], X[j], 'sine')
            
    return Distances

def cluster_purity(X, centers, opt_type, labels_true):
    #calculate distance matrix
    d_mat = distance_matrix(X, centers, opt_type)

    #find the closest center for each point
    index = np.argmin(d_mat, axis = 0)
    
    count = 0
    for i in range(len(centers)):
        idx = np.where(index == i)[0]
        if len(idx) != 0:
            cluster_labels = [labels_true[i] for i in idx]
            most_common_label = max(set(cluster_labels), key = cluster_labels.count)
            # count += cluster_labels.count(most_common_label)
            count += cluster_labels.count(most_common_label)/len(idx)

    # return count/len(X)
    return count/len(centers)


def lbg_subspace(X, epsilon, n_centers = 17, opt_type = 'sine', n_its = 10, seed = 1):
    n_pts = len(X)
    error = 1
    r = 48
    distortions = []

    #init centers
    np.random.seed(seed)
    centers = []
    for i in range(n_centers):
        centers.append(X[np.random.randint(n_pts)])

    #calculate distance matrix
    d_mat = distance_matrix(X, centers, opt_type)

    #find the closest center for each point
    index = np.argmin(d_mat, axis = 0)

    #calculate first distortion
    new_distortion = np.sum(d_mat[index])

    distortions.append(new_distortion)


    errors = []
    while error > epsilon:

        #set new distortion as old one
        old_distortion = new_distortion

        m = len(centers)

        #calculate new centers
        centers = []
        for c in range(m):
            idx = np.where(index == c)[0]
            if len(idx) > 0:
                if opt_type == 'sinesq':
                    centers.append(ca.flag_mean([X[i] for i in idx], r))
                elif opt_type == 'l2_med':
                    centers.append(ca.l2_median([X[i] for i in idx], .1, r, 1000)[0])
                else:
                    centers.append(ca.irls_flag([X[i] for i in idx], r, n_its, opt_type, opt_type)[0])
        #         centers.append(np.mean([X[i] for i in idx], axis = 0))

        #calculate distance matrix
        d_mat = distance_matrix(X, centers, opt_type)

        #find the closest center for each point
        index = np.argmin(d_mat, axis = 0)

        #new distortion
        new_distortion = np.sum(d_mat[index])

        distortions.append(new_distortion)

        if new_distortion <0.00000000001:
            error = 0
        else:
            error = np.abs(new_distortion - old_distortion)/old_distortion
        errors.append(error)
        print(error)

    return centers, errors, distortions





n_its= 10
seed = 0
n_trials = 10


base_path = './data/action_youtube_gr/'
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
        centers_sin, error_sin, dist_sin = lbg_subspace(X, .0001, n_centers = n, opt_type = 'sine', n_its = 10, seed = trial)
        sin_purity = cluster_purity(X, centers_sin, 'sine', labels_true)
        print(len(error_sin))
        print('flg start')
        centers_flg, error_flg, dist_flg = lbg_subspace(X, .0001, n_centers = n, opt_type = 'sinesq', seed = trial)
        flg_purity = cluster_purity(X, centers_flg, 'sinesq', labels_true)


        Purities = Purities.append({'Algorithm': 'Flag Median', 
                                'Codebook Size': n,
                                'Cluster Purity': sin_purity},
                                ignore_index = True)

        Purities = Purities.append({'Algorithm': 'Flag Mean', 
                                'Codebook Size': n,
                                'Cluster Purity': flg_purity},
                                ignore_index = True)
    # print(Purities)
    # Purities.to_csv('youtube_LBG_results_20trials'+str(n)+'.csv')
        
# Purities.to_csv('LBG_results_20trials.csv')

sns.boxplot(x='Codebook Size', y='Cluster Purity', hue='Algorithm', data = Purities)
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.savefig(f_name, bbox_inches='tight')


