import numpy as onp
from rockpool.networks import NetworkADS
from rockpool.timeseries import TSContinuous
from Utils import generate_filtered_noise
import matplotlib.pyplot as plt


Nc = 128
num_neurons = 300
Nb = num_neurons
N_out = 1

print("Building network with N: %d Nc: %d and Nb: %d" % (num_neurons,Nc,Nb))

lambda_d = 20
lambda_v = 20
tau_membrane = 1/ lambda_v

tau_slow = 0.05
tau_out = 0.05

tau_syn_fast = 1e-3
mu = 0.002
nu = 0.001
D = onp.random.randn(Nc,num_neurons) / Nc
weights_in = D
weights_out = D.T
weights_fast = (D.T@D + mu*lambda_d**2*onp.eye(num_neurons)) 
weights_slow = onp.random.randn(Nb,num_neurons)
eta = 0.01
k = 500
noise_std = 0.0
# - Pull out thresholds
v_thresh = (nu * lambda_d + mu * lambda_d**2 + onp.sum(abs(D.T), -1, keepdims = True)**2) / 2
v_reset = v_thresh - onp.reshape(onp.diag(weights_fast), (-1, 1))
v_rest = v_reset
# - Fill the diagonal with zeros
onp.fill_diagonal(weights_fast, 0)
dt = 0.001

net = NetworkADS.SpecifyNetwork(N=num_neurons,
                                Nc=Nc,
                                Nb=Nb,
                                weights_in=weights_in * tau_membrane,
                                weights_out= weights_out,
                                weights_fast= - weights_fast * tau_membrane / 1e-3,
                                weights_slow=weights_slow * tau_membrane,
                                eta=eta,
                                k=k,
                                noise_std=noise_std,
                                dt=dt,
                                v_thresh=v_thresh,
                                v_reset=v_reset,
                                v_rest=v_rest,
                                tau_mem=tau_membrane,
                                tau_syn_r_fast=tau_syn_fast,
                                tau_syn_r_slow=tau_slow,
                                tau_syn_r_out=tau_out,
                                record=True,
                                )

data = generate_filtered_noise(Nc, total_duration=5.0, dt=dt,sigma=30,plot=False)[:,:-1]
target = data + 0.1*onp.random.randn(data.shape[0],data.shape[1])
data *= 1000

def do_training():
    time_base = onp.arange(0,5.0,dt)
    ts_data = TSContinuous(time_base, data.T)
    ts_target = TSContinuous(time_base, target.T)

    # plt.subplot(211)
    # ts_data.plot()
    # plt.subplot(212)
    # ts_target.plot()
    # plt.show()

    net.lyrRes.is_training = True
    net.lyrRes.ts_target = ts_target
    sim = net.evolve(ts_input=ts_data, verbose=False)
    net.reset_all()

    # sim["lyrRes"].plot(); plt.show()


do_training()