import numpy as np
import center_algorithms as ca
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.manifold import MDS

import math



def get_minor(arr, n_1, n_2, m_1, m_2):
    return np.linalg.det(np.vstack([arr[n_1,[m_1,m_2]],arr[n_2,[m_1,m_2]]]))

def pluker_embedding(arr, n=4, k=2):

    embedded_pt = np.zeros((math.comb(n,2)*math.comb(k,2),1))
    idx = 0
    for i in range(n):
        for ii in range(i+1, n):
            for j in range(k):
                for jj in range(j+1, k):
                    embedded_pt[idx] = get_minor(arr,i,ii,j,jj)
                    idx +=1

    embedded_pt = embedded_pt/np.linalg.norm(embedded_pt)
    return embedded_pt

'''
This function verifies that we found a local optimizer for
the sine median or maximum cosine problem.
It does this by checking 100 points around the optimizer.

Inputs: 
    optimizer- a numpy array that represente the suggested optimizer
    opf_fn- a string, 'sine' for sine median and 'cosine' for maximum cosine
    data- a list of numpy arrays representing points on grassmannians
Outputs:
    local optimizer- True if all checked points rsesult in objective function values
                        that are greater than the Sine Median problem or smaller than
                        the Maximum Cosine value
'''
def objective_function_values(data, center_rep, n_random_pts, r = 5, n = 20):
    #checking random points
    errs = []
    random_points = []
    for i in range(n_random_pts):        
        #random points between -.5 and .5 times .01
        random_point_raw = center_rep + (np.random.rand(n,r)-.5)*.5
        random_point = np.linalg.qr(random_point_raw)[0][:,:r]
        random_points.append(random_point)
    
    
    sinsq_err = []
    sin_err = []
    geo_err = []
    cos_err = []
    for Y in random_points:
    #     print(Y.shape)
        sinsq_err.append(ca.calc_error_1_2(data, Y, 'sinesq'))
        sin_err.append(ca.calc_error_1_2(data, Y, 'sine'))
        geo_err.append(ca.calc_error_1_2(data, Y, 'l2_med'))
        cos_err.append(ca.calc_error_1_2(data, Y, 'cosine'))
    
    errs = {'Flag Mean': sinsq_err,
            'Flag Median': sin_err,
            'L2 Median': geo_err,
            'Max Cor Flag': cos_err}

    return errs, random_points

def generate_data(n_random_pts = 5000):
    k=2 #Gr(k1,n)
    r=2
    n=4
    num_points = 100 #number of points in dataset
    n_outliers = 30
    
    # n_random_pts = 0

    np.random.seed(0)

    center = np.random.rand(n,k)*10
    center_rep = np.linalg.qr(center)[0][:,:k]

    outlier_center = np.random.rand(n,k)*10
    outlier_center_rep = np.linalg.qr(outlier_center)[0][:,:k]

    #generate dataset of points in Gr(k,n)
    print('generating data')
    print('.\n.\n.')
    gr_list = []
    for i in range(num_points-n_outliers):
        Y_raw = center_rep + (np.random.rand(n,k)-.5)*.05
        Y = np.linalg.qr(Y_raw)[0][:,:k]
        gr_list.append(Y)

    for i in range(n_outliers):
        Y_raw = outlier_center_rep + (np.random.rand(n,k)-.5)*.05
        Y = np.linalg.qr(Y_raw)[0][:,:k]
        gr_list.append(Y)

    center_clusters_rep = (center_rep + outlier_center)/2
    center_clusters_rep = center_clusters_rep/np.linalg.norm(center_clusters_rep)

    print('----------------')
    print('calculating objective function values')
    print('.\n.\n.')
    errs, random_points = objective_function_values(gr_list, center_clusters_rep, n_random_pts, r, n)

    print('----------------')
    print('calculating central prototypes')
    print('.\n.\n.')
    central_prototype_ids = {}
    for prototype_name in errs:
        print(prototype_name)
        if prototype_name == 'Max Cor Flag':
            best_idx = np.argmax(errs[prototype_name ])
        else:
            best_idx = np.argmin(errs[prototype_name ])
        central_prototype_ids[prototype_name] = best_idx
    print('prototype ids are: ')
    print(central_prototype_ids)

    # the_prototypes = [random_points[ii] for _, ii in central_prototype_ids.items()]

    for name in central_prototype_ids.items():
        print(name)


    all_points = random_points + gr_list
    # all_points = the_prototypes + gr_list

    return all_points, central_prototype_ids, errs

def generate_pluker_distance(all_points):
    print('----------------')
    print('calculating pluker distance matrix')
    print('.\n.\n.')
    all_pluker_points = []
    for pt in all_points:
        all_pluker_points.append(pluker_embedding(pt))

    pluker_array = np.hstack(all_pluker_points)


    ssq = 1 - pluker_array.T @ pluker_array
    ssq[np.where(ssq <0)] = 0
    pluker_D = np.sqrt(ssq)

    return pluker_D

def generate_chordal_distance(all_points):
    print('----------------')
    print('calculating distance matrix')
    print('.\n.\n.')
    total_p = len(all_points)
    D = np.zeros((total_p, total_p))
    for i in range(total_p):
        print(f'point {i} finished')
        for j in range(i+1,total_p):
            D[i,j] = ca.calc_error_1_2([all_points[i]], all_points[j], sin_cos = 'sine')
            D[j,i] = D[i,j].copy()
    return D

def run_mds(D):
    embedding = MDS(n_components=2, dissimilarity='precomputed')
    X_transformed = embedding.fit_transform(D)
    return X_transformed

def plot_results(save_suffix, errs, X_transformed, n_random_pts):

    colors = {'Flag Median':'tab:blue', 'Max Cor Flag':'tab:red', 'L2 Median':'tab:orange', 'Flag Mean':'tab:green'}
    prototype_names = {'Flag Median':0, 'Max Cor Flag':1, 'L2 Median':2, 'Flag Mean':3}
    markers = {'Flag Median':'s', 'Max Cor Flag':'o', 'L2 Median':'>', 'Flag Mean':'<'}
    plt.figure()
    # plt.scatter(X_transformed[:n_random_pts,0], X_transformed[:n_random_pts,1], c =errs)
    plt.scatter(X_transformed[n_random_pts:,0], X_transformed[n_random_pts:,1], c = 'k', marker = 'x', s=8)
    for prototype_name in {'Flag Median', 'Max Cor Flag', 'L2 Median', 'Flag Mean'}:
        plt.scatter(X_transformed[central_prototype_ids[prototype_name],0], X_transformed[central_prototype_ids[prototype_name],1], c = colors[prototype_name], label = prototype_name, marker = markers[prototype_name])


    plt.xlabel('MDS 1')
    plt.ylabel('MDS 2')
    plt.legend()
    plt.savefig(f'../results/gr2_4_{save_suffix}.png')


    colors = {'Flag Median':'tab:blue', 'Max Cor Flag':'tab:red', 'L2 Median':'tab:orange', 'Flag Mean':'tab:green'}
    # prototype_names = {'Flag Median':0, 'Max Cor Flag':1, 'L2 Median':2, 'Flag Mean':3}
    markers = {'Flag Median':'s', 'Max Cor Flag':'o', 'L2 Median':'>', 'Flag Mean':'<'}


    for prototype in ['Flag Median', 'Max Cor Flag', 'L2 Median', 'Flag Mean']:
        plt.figure()
        plt.scatter(X_transformed[:n_random_pts,0], X_transformed[:n_random_pts,1], c =errs[prototype], s = 6)
        plt.scatter(X_transformed[n_random_pts:,0], X_transformed[n_random_pts:,1], c = 'k', marker = 'x', s = 8)
        for prototype_name in {'Flag Median', 'Max Cor Flag', 'L2 Median', 'Flag Mean'}:
            plt.scatter(X_transformed[central_prototype_ids[prototype_name],0], X_transformed[central_prototype_ids[prototype_name],1], c = colors[prototype_name], label = prototype_name, marker = markers[prototype_name])
        plt.colorbar()
        plt.xlabel('MDS 1')
        plt.ylabel('MDS 2')
        plt.title(prototype)
        plt.legend()
        save_prototype_name = '_'.join(prototype.split(' '))
        plt.savefig(f'../results/gr2_4_{save_prototype_name}_{save_suffix}.png')




if __name__ == '__main__':
    n_random_pts = 5000

    all_points, central_prototype_ids, errs = generate_data(n_random_pts)

    pluker_D = generate_pluker_distance(all_points)

    pluker_mds = run_mds(pluker_D)

    plot_results('pluker', errs, pluker_mds, n_random_pts)

    chordal_D = generate_chordal_distance(all_points)

    chordal_mds = run_mds(chordal_D)

    plot_results('chordal', errs, chordal_mds, n_random_pts)







