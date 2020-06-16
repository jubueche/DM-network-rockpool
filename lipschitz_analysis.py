import warnings
warnings.filterwarnings('ignore')
import json
import numpy as np
import matplotlib
matplotlib.rc('font', family='Times New Roman')
matplotlib.rc('text')
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt
from SIMMBA import BaseModel
from SIMMBA.experiments.HeySnipsDEMAND import HeySnipsDEMAND
from SIMMBA import BatchResult
from rockpool.timeseries import TSContinuous
from rockpool import layers
from rockpool.layers import ButterMelFilter, RecRateEulerJax_IO, H_tanh
from rockpool.layers.training import add_shim_rate_jax_sgd
from rockpool.networks import NetworkADS
from sklearn import metrics
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import argparse
from Utils import plot_matrices, filter_1d, generate_xor_sample, approximate_lipschitzness_temporal_xor
from copy import deepcopy

"""
@brief  - This script explores the correlation between accuracy drops caused by mismatch and the Lipschitz constant for each classifier.
          To get a statistically sound estimation of both, the accuracy drop and the Lipschitz constant, the experiment is repeated a total of
          N=10 times. We hope to show that by increasing the number of spiking neurons, and more importantly, the number of rate neurons
          the Lip. constant and also the testing accuracy drop is significantly reduced.
        - We apply mismatch to the spiking thresholds, membrane time constants and synaptic time constants since there mismatch has the biggest effect.
"""

def get_data(data, duration, dt, rate_layer, amplitude):
    time_base = np.arange(0.0,duration,dt)
    ts_data = TSContinuous(time_base, data)
    # - Pass through the rate network
    ts_rate_out = rate_layer.evolve(ts_data)
    rate_layer.reset_all()
    # - Get the target dynamics
    ts_rate_net_target_dynamics = rate_layer.res_acts_last_evolution
    # - Get the input into the spiking network
    ts_spiking_in = TSContinuous(rate_layer.res_inputs_last_evolution.times,amplitude*rate_layer.res_inputs_last_evolution.samples)
    return (ts_spiking_in, ts_rate_net_target_dynamics, ts_rate_out)

def get_test_accuracy(ads_net, w_in, w_out, data, N_test = 100):
    correct = 0
    for (ts_spiking_in, tgt_label) in data:
        ads_net.lyrRes.ts_target = ts_spiking_in
        val_sim = ads_net.evolve(ts_input=ts_spiking_in, verbose=False)
        out_val = val_sim["output_layer"].samples.T
        ads_net.reset_all()
        final_out = out_val.T @ w_out
        final_out = filter_1d(final_out, alpha=0.99)
        if(final_out[np.argmax(np.abs(final_out))] > 0):
            predicted_label = 1
        else:
            predicted_label = 0
        if(predicted_label == tgt_label):
            correct += 1
    return correct / len(data)

def apply_mismatch(ads_net, mismatch_std=0.2):
    mismatch_ads_net = deepcopy(ads_net)
    N = mismatch_ads_net.lyrRes.weights.shape[0]
    mismatch_ads_net.lyrRes.tau_syn_r_slow = np.abs(np.random.randn(N)*mismatch_std*np.mean(ads_net.lyrRes.tau_syn_r_slow) + np.mean(ads_net.lyrRes.tau_syn_r_slow))
    mismatch_ads_net.lyrRes.tau_mem = np.abs(np.random.randn(N)*mismatch_std*np.mean(ads_net.lyrRes.tau_mem) + np.mean(ads_net.lyrRes.tau_mem))
    mismatch_ads_net.lyrRes.v_thresh = np.abs(np.random.randn(N)*mismatch_std*np.mean(ads_net.lyrRes.v_thresh) + np.mean(ads_net.lyrRes.v_thresh))
    return mismatch_ads_net


base_path = "/home/julian/Documents/dm-network-rockpool/Resources/temporal-xor/"
# - Rate network paths
rate_net_64_path = os.path.join(base_path,"temporal_xor_rate_model_longer_target.json")
rate_net_128_path = os.path.join(base_path,"temporal_xor_rate_model_128.json")

# - Load both rate networks
with open(rate_net_64_path, "r") as f:
    config_64 = json.load(f)
with open(rate_net_128_path, "r") as f:
    config_128 = json.load(f)

rate_layer_64 = RecRateEulerJax_IO(w_in=config_64["w_in"],
                                        w_recurrent=config_64["w_recurrent"],
                                        w_out=config_64["w_out"],
                                        tau=config_64["tau"],
                                        bias=config_64["bias"],
                                        activation_func=H_tanh,
                                        dt=config_64["dt"],
                                        noise_std=config_64["noise_std"],
                                        name="hidden_64")

rate_layer_128 = RecRateEulerJax_IO(w_in=config_128["w_in"],
                                        w_recurrent=config_128["w_recurrent"],
                                        w_out=config_128["w_out"],
                                        tau=config_128["tau"],
                                        bias=config_128["bias"],
                                        activation_func=H_tanh,
                                        dt=config_128["dt"],
                                        noise_std=config_128["noise_std"],
                                        name="hidden_128")

# - Spiking model paths
spiking_384_64 = os.path.join(base_path, "node_14967141530_test_acc0.9866666666666667threshold0.7eta0.001val_acc1.0tau_slow0.1tau_out0.1num_neurons384num_dist_weights-1.json")
spiking_768_64 = os.path.join(base_path, "node_14967141530_test_acc0.99threshold0.7eta0.001val_acc1.0tau_slow0.1tau_out0.1num_neurons768num_dist_weights-1.json")
spiking_1024_64 = os.path.join(base_path, "node_124989100848_test_acc1.0threshold0.7eta0.0001val_acc1.0tau_slow0.07tau_out0.07num_neurons1024num_dist_weights-1.json")
spiking_1536_64 = os.path.join(base_path, "node_14362694660_test_acc0.98threshold0.7eta0.001val_acc1.0tau_slow0.07tau_out0.07num_neurons1536num_dist_weights-1.json")

spiking_384_128 = os.path.join(base_path, "node_19299043247_test_acc1.0threshold0.7eta0.0001val_acc1.0tau_slow0.07tau_out0.07num_neurons384num_dist_weights-1.json")
spiking_768_128 = os.path.join(base_path,"node_18832352344tmpnum_768_128rate_96_val.json")
spiking_1024_128 = os.path.join(base_path, "node_12251279007_test_acc1.0threshold0.7eta0.0001val_acc1.0tau_slow0.07tau_out0.07num_neurons1024num_dist_weights-1.json")
spiking_1536_128 = os.path.join(base_path, "node_18379347811_test_acc0.95threshold0.7eta0.0001val_acc0.93tau_slow0.1tau_out0.1num_neurons1534num_dist_weights-1.json")

spiking_models_paths = [spiking_384_64, spiking_768_64, spiking_1024_64, spiking_1536_64, spiking_384_128, spiking_768_128, spiking_1024_128, spiking_1536_128]

duration = 1.0
dt = 0.001
N_test = 300
amplitude = 10 / 0.05

# - Cache some data. Expected to be in format [(ts_spiking_in, tgt_label)]
data_64 = []
data_128 = []
for i in range(N_test):
    raw_data, target, _ = generate_xor_sample(total_duration=duration, dt=dt, amplitude=1.0)
    (ts_spiking_in_64, _, _) = get_data(raw_data, duration=duration, dt=dt, rate_layer=rate_layer_64, amplitude=amplitude)
    (ts_spiking_in_128, _, _) = get_data(raw_data, duration=duration, dt=dt, rate_layer=rate_layer_128, amplitude=amplitude)
    if((target > 0.5).any()):
        tgt_label = 1
    else:
        tgt_label = 0
    data_64.append((ts_spiking_in_64, tgt_label))
    data_128.append((ts_spiking_in_128, tgt_label))

for path in spiking_models_paths:

    net = NetworkADS.load(path)
    Nc = net.lyrRes.weights_in.shape[0]
    if(Nc == 64):
        rate_net = rate_layer_64
        data = data_64
    else:
        rate_net = rate_layer_128
        data = data_128

    N = net.lyrRes.weights_fast.shape[0]

    # - Get testing accuracy
    test_accuracy = get_test_accuracy(net, rate_net.w_in, rate_net.w_out, data=data, N_test=N_test)

    print(f"[{N},{Nc}] Test accuracy is {test_accuracy}")

    # - Do the same for the mismatched network
    # - We apply random mismatch for 10 iterations
    test_accuracies_mismatch = []
    for _ in range(30):
        mismatch_net = apply_mismatch(net, mismatch_std=0.2)
        test_accuracies_mismatch.append(get_test_accuracy(mismatch_net, rate_net.w_in, rate_net.w_out, data=data, N_test=100))

    print(f"[{N},{Nc}] Mismatch: Mean test accuracy is {np.mean(test_accuracies_mismatch)} and std is {np.std(test_accuracies_mismatch)}")

    # - At last calculate the Lipschitz constant. Also repeat this 10 times
    lipschitz_N = 10
    lipschitz_X = []
    lipschitznesses = []
    for i in range(10):
        data_tmp = data[i*10:i*10+10]
        lipschitz_X = [ts_spiking_in for (ts_spiking_in, _) in data_tmp]
        lipschitzness = approximate_lipschitzness_temporal_xor(lipschitz_X, net, rate_net.w_out, perturb_steps=10, mismatch_std=0.2)
        lipschitznesses.append(lipschitzness)
    
    print(f"[{N},{Nc}] Mean Lipschitz constant is {np.nanmean(lipschitznesses)} and std is {np.nanstd(lipschitznesses)}")
