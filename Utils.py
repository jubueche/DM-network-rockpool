import numpy as np
import sys
import os
from scipy import interpolate
import matplotlib.pyplot as plt

    
def generate_xor_sample(total_duration, dt, amplitude=1, use_smooth=True, plot=False):
    """
    Generates a temporal XOR signal
    """
    input_duration = 2/3*total_duration
    # Create a time base
    t = np.linspace(0,total_duration, int(total_duration/dt)+1)
    
    first_duration = np.random.uniform(low=input_duration/10, high=input_duration/4 )
    second_duration = np.random.uniform(low=input_duration/10, high=input_duration/4 )

    end_first = np.random.uniform(low=first_duration, high=2/3*input_duration-second_duration)
    start_first = end_first - first_duration

    start_second = np.random.uniform(low=end_first + 0.1, high=2/3*input_duration-second_duration) # At least 200 ms break
    end_second = start_second+second_duration

    data = np.zeros(int(total_duration/dt)+1)

    i1 = np.random.rand() > 0.5
    i2 = np.random.rand() > 0.5
    response = (((not i1) and i2) or (i1 and (not i2)))
    if(i1):
        a1 = 1
    else:
        a1 = -1
    if(i2):
        a2 = 1
    else:
        a2 = -1
    data[(start_first <= t) & (t < end_first)] = a1
    data[(start_second <= t) & (t < end_second)] = a2

    if(use_smooth):
        sigma = 10
        w = (1/(sigma*np.sqrt(2*np.pi)))* np.exp(-((np.linspace(1,1000,int(1/dt))-500)**2)/(2*sigma**2))
        w = w / np.sum(w)
        data = amplitude*np.convolve(data, w, "same")
    else:
        data *= amplitude

    target = np.zeros(int(total_duration/dt)+1)
    if(response):
        ar = 1.0
    else:
        ar = -1.0
    
    target[int(1/dt*(end_second+0.2))] = ar
    sigma = 20
    w = (1/(sigma*np.sqrt(2*np.pi)))* np.exp(-((np.linspace(1,1000,int(1/dt))-500)**2)/(2*sigma**2))
    w = w / np.sum(w)
    target = np.convolve(target, w, "same")
    target /= np.max(np.abs(target))

    if(plot):
        eps = 0.05
        plt.subplot(211)
        plt.plot(t, data)
        plt.ylim([-amplitude-eps, amplitude+eps])
        plt.subplot(212)
        plt.plot(t, target)
        plt.show()

    return (data, target)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0) 
    return (cumsum[N:,:] - cumsum[:-N,:]) / float(N)

def pISI_variance(sim_result):
    """
    Compute the variance of the population inter-spike intervals
    Parameters:
        sim_result : Object of type evolution Object that was returned after having called evolve
    Returns:
        variance of difference array (variance of pISI) in milliseconds
    """
    times_c = sim_result['lyrRes'].times[sim_result['lyrRes'].channels > -1]
    np.sort(times_c) # Sorts in ascending order
    diff = np.diff(times_c)
    return np.sqrt(np.var(diff * 1000))

def generate_filtered_noise(N, total_duration, dt, sigma=30, plot=False):
    w = (1/(sigma*np.sqrt(2*np.pi)))* np.exp(-((np.linspace(1,1000,int(1/dt))-500)**2)/(2*sigma**2))
    w = w / np.sum(w)
    # Get the rate vectors
    rateVectors = (np.random.multivariate_normal(np.zeros(N), np.eye(N), int(total_duration / dt)+1)).T
    for d in range(N):
        rateVectors[d,:] = np.convolve(rateVectors[d,:], w, 'same')
        # Scale tp [-1,1]
        rateVectors[d,:] = 2*((rateVectors[d,:] - np.min(rateVectors[d,:])) / (np.max(rateVectors[d,:]) - np.min(rateVectors[d,:])))-1

    if(plot):
        plt.plot(rateVectors[0:2,:].T)
        plt.show()

    return rateVectors

def my_max(vec):
    k = np.argmax(vec)
    m = vec[k]
    return (m,k)

def get_input(A, Nx, TimeL, w):
    InputL = A*(np.random.multivariate_normal(np.zeros(Nx), np.eye(Nx), TimeL)).T
    for d in range(Nx):
        InputL[d,:] = np.convolve(InputL[d,:], w, 'same')
    return InputL


def get_matrix_distance(A,B):
    optscale = np.trace(np.matmul(A.T, B)) / np.sum(B**2)
    Cnorm = np.sum(A**2)
    ErrorC = np.sum(np.sum((A - optscale*B)**2 ,axis=0)) / Cnorm
    return ErrorC

def plot_matrices(A,B,C=None, title_A=None, title_B=None, title_C=None):
    if(C is not None):
        base = 130
    else:
        base = 120
    plt.subplot(base +1)
    plt.title(title_A)
    im = plt.matshow(A, fignum=False, cmap="RdBu")
    plt.xticks([], [])
    plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.subplot(base+2)
    plt.title(title_B)
    im = plt.matshow(B, fignum=False, cmap="RdBu")
    plt.xticks([], [])
    plt.colorbar(im,fraction=0.046, pad=0.04)
    if(C is not None):
        plt.subplot(base+3)
        plt.title(title_C)
        im = plt.matshow(C, fignum=False, cmap="RdBu")
        plt.xticks([], [])
        plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def round_to_nearest(x,l,r):
    if(round((x-l)/(r-l)) == 0):
        return l
    else:
        return r

def discretize(Omega, bin_edges):
    C = np.copy(Omega)
    diag_C = np.copy(np.diagonal(C))
    np.fill_diagonal(C,0)
    C_flat = C.ravel()
    indices = np.digitize(C_flat, bins = bin_edges, right = True)
    n = len(bin_edges)
    for idx,i in enumerate(indices):
        if(i >= n):
            i = n-1
        C_flat[idx] = round_to_nearest(C_flat[idx], bin_edges[i-1], bin_edges[i])
    C = C_flat.reshape(C.shape)
    np.fill_diagonal(C, diag_C)
    return C

def discretize_F(F, bin_edges):
    F = np.copy(F)
    F_flat = F.ravel()
    indices = np.digitize(F_flat, bins = bin_edges, right = True)
    n = len(bin_edges)
    for idx,i in enumerate(indices):
        if(i >= n):
            i = n-1
        F_flat[idx] = round_to_nearest(F_flat[idx], bin_edges[i-1], bin_edges[i])
    F = F_flat.reshape(F.shape)
    return F


def bin_omega(C_real, F_disc, max_syn_per_neuron, debug=False):
    
    assert(np.diagonal(C_real) == 0).all(), "Please set the diagonal to zero"
        
    C_new_discrete = np.zeros(C_real.shape)

    _, bin_edges = np.histogram(C_real.reshape((-1,1)), bins = 2*max_syn_per_neuron, range=(np.min(C_real),np.max(C_real)))
    C_new_discrete = np.digitize(C_real.ravel(), bins = bin_edges, right = True).reshape(C_new_discrete.shape) - max_syn_per_neuron
    
    assert (C_new_discrete <= max_syn_per_neuron).all() and (C_new_discrete >= -max_syn_per_neuron).all(), "Error, have value > or < than max/min in Omega"
                
    number_available_per_neuron = 64 - np.sum(np.abs(F_disc), axis=0)

    if(not (((number_available_per_neuron - np.sum(np.abs(C_new_discrete), axis=1)) >= 0).all())):
        # - Reduce the number of weights here, if necessary

        for idx in range(C_new_discrete.shape[0]):
            num_available = number_available_per_neuron[idx]

            # - Use sorting + cutoff to keep the most dominant weights
            sorted_indices = np.flip(np.argsort(np.abs(C_new_discrete[idx,:])))
            sub_sum = 0; i = 0
            while(sub_sum < num_available):
                if(i == len(sorted_indices)):
                    break
                sub_sum += np.abs(C_new_discrete[idx,:])[sorted_indices[i]]
                i += 1
            tmp = np.zeros(len(sorted_indices))
            tmp[sorted_indices[0:i-1]] = C_new_discrete[idx,sorted_indices[0:i-1]]
            C_new_discrete[idx,:] = tmp

    assert ((number_available_per_neuron - np.sum(np.abs(C_new_discrete), axis=1)) >= 0).all(), "More synapses used than available"
    if(debug):
        print("Number of neurons used: %d / %d" % (np.sum(np.abs(C_new_discrete)), np.sum(number_available_per_neuron)))

    return C_new_discrete

def bin_F(F_real, max_syn_per_neuron):
    F_new_discrete = np.zeros(F_real.shape)
    _, bin_edges = np.histogram(F_real.reshape((-1,1)), bins = 2*max_syn_per_neuron, range=(np.min(F_real),np.max(F_real)))
    F_new_discrete = np.digitize(F_real.ravel(), bins = bin_edges, right = True).reshape(F_new_discrete.shape) - max_syn_per_neuron

    return F_new_discrete

def filter_1d(data, alpha = 0.9):
    last = data[0]
    out = np.zeros((len(data),))
    out[0] = last
    for i in range(1,len(data)):
        out[i] = alpha*out[i-1] + (1-alpha)*data[i]
        last = data[i]
    return out