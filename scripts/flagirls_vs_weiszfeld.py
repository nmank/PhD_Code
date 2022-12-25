import numpy as np
import sys
sys.path.append('./scripts/')
import center_algorithms as ca
import matplotlib.pyplot as plt
import torch
import pandas
import seaborn as sns
from sklearn.manifold import MDS
import time


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
def sanity_check(optimizer, opt_fn, data):
    
    n,r = optimizer.shape
    
    
    #objective function value for suggested optimizer
    sln_val = ca.calc_error_1_2(data, optimizer, opt_fn) 
    
    #stays true if optimizer is a local min
    local_optimizer = True

    #checking random points
    perturb_vals = []
    for i in range(100):
        
        #random points between -.5 and .5 times .00001
        perturb = (np.random.rand(n,r)-5)*.00001 
        perturb_check = np.linalg.qr(perturb + optimizer)[0][:,:r]
        
        #check objective function value
        perturb_vals.append(ca.calc_error_1_2(data, perturb_check, opt_fn))

        if opt_fn == 'sine' or opt_fn == 'l2_med':
            if perturb_vals[i] < sln_val:
                local_optimizer = False
        elif opt_fn == 'cosine':
            if perturb_vals[i] > sln_val:
                local_optimizer = False
        

    # if not local_optimizer:
    #     print('Algorithm did not converge to maximizer')
    return local_optimizer

def calc_chordal_dist(X, Y, r):
    sum_sin = r- np.trace(X.T @ Y @ Y.T @ X)
    if sum_sin < 0:
        sum_sin = 0
    return np.sqrt(sum_sin)

def calc_cos_sim(X, Y, r):
    sum_cos = np.trace(X.T @ Y @ Y.T @ X)
    if sum_cos < 0:
        sum_cos = 0
    return np.sqrt(sum_cos)


if __name__ == '__main__':
    k=6 #Gr(k1,n)
    r=6
    n=100
    n_its = 1000 #number of iterationss
    n_trials = 20


    num_points = 200 #number of points in dataset
    n_trials = 20 #number of trials for sanity check

    len_flag_med_errs = []
    len_max_corr_errs = []
    len_l2_errs_data = [] 
    len_l2_errs_rand = []


    flag_med_time = []
    max_corr_time = []
    l2_errs_data_time = []
    l2_errs_rand_time = []
    for seed in range(n_trials):

        np.random.seed(seed+1)

        center = np.random.rand(n,k)*10
        center_rep = np.linalg.qr(center)[0][:,:k]

        diameter = 10**(-2)
        #generate dataset of points in Gr(k,n)
        gr_list = []
        for i in range(num_points):
            Y_raw = center_rep + (np.random.rand(n,k)-.5)*diameter
            Y = np.linalg.qr(Y_raw)[0][:,:k]
            gr_list.append(Y)


        #calculate sine median
        start = time.time()
        sine_median_error = ca.irls_flag(gr_list, r, n_its-1, 'sine', opt_err = 'sine', seed = seed)[1]
        flag_med_time.append(time.time()-start)
        len_flag_med_errs.append(len(sine_median_error))

        # #calc maximally correlated flag
        # start = time.time()
        # max_corr_error = ca.irls_flag(gr_list, r, n_its-1, 'cosine', opt_err = 'cosine', seed = seed)[1]
        # max_corr_time.append(time.time()-start)
        # len_max_corr_errs.append(len(max_corr_error))

        #calc l2 median datapoint initialization
        start = time.time()
        l2_errs_data = ca.l2_median(gr_list, .1, r, n_its, seed, True)[1]
        l2_errs_data_time.append(time.time()-start)
        len_l2_errs_data.append(len(l2_errs_data))

        #calc l2 median random initialization
        start = time.time()
        l2_errs_rand = ca.l2_median(gr_list, .1, r, n_its, seed, False)[1]
        l2_errs_rand_time.append(time.time()-start)
        len_l2_errs_rand.append(len(l2_errs_rand))
    

    print('average number of its to converge')
    print('flag median: '+str(np.mean(len_flag_med_errs))+' +/-'+str(np.std(len_flag_med_errs)))
    # print('maximally correlated flag: '+str(np.mean(len_max_corr_errs))+' +/-'+str(np.std(len_max_corr_errs)))
    print('l2 median: (init datapoint) '+str(np.mean(len_l2_errs_data))+' +/-'+str(np.std(len_l2_errs_data)))
    print('l2 median: (init same as flag median) '+str(np.mean(len_l2_errs_rand))+' +/-'+str(np.std(len_l2_errs_rand)))

            
    print('average times to convergence')   
    print('flag median: '+str(np.mean(flag_med_time))+' +/-'+str(np.std(flag_med_time)))
    # print('maximally correlated flag: '+str(np.mean(max_corr_time))+' +/-'+str(np.std(max_corr_time)))
    print('l2 median: (init datapoint) '+str(np.mean(l2_errs_data_time))+' +/-'+str(np.std(l2_errs_data_time)))
    print('l2 median: (init same as flag median) '+str(np.mean(l2_errs_rand_time))+' +/-'+str(np.std(l2_errs_rand_time)))