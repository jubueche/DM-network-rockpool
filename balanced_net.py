# - Script to tune parameters for the balanced network
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
import matplotlib.pyplot as plt
from Utils import generate_filtered_noise
from rockpool.networks import NetworkADS
from rockpool.timeseries import TSContinuous

# - Input function for encoding a smooth signal to rates
def get_input(Nc=2,duration=1.0, dt=1e-4,sigma=30,plot=False):
    filtered_noise = 0.5*(1+generate_filtered_noise(N=Nc, total_duration=duration, dt=dt, sigma=sigma, plot=False))
    if(plot):
        plt.figure(figsize=(10,3)) ; plt.plot(np.linspace(0,duration,int(duration/dt)+1), filtered_noise.T)
        plt.show()
    return filtered_noise

np.random.seed(42)

tmp = [5,10,14,20]

Nc = 2
duration = 1.0
dt=1e-3

# - Generate input signals
inp = get_input(Nc=Nc, duration=duration, dt=dt, sigma=30, plot=False)[:,:int(duration/dt)].T

for x in tmp:
    print("X is",x)

    lambda_d = 1 / 0.07
    lambda_v = 1 / 0.07
    tau_mem = 1/ lambda_v
    amplitude = 50 / tau_mem

    inp *= amplitude

    num_neurons = 50

    tau_slow = 0.07
    tau_out = 0.07

    tau_syn_fast = 0.07
    mu = 0.001
    nu = 0.0001
    D = np.random.randn(Nc,num_neurons) / Nc
    # weights_in = D
    # weights_out = D.T
    weights_fast = (D.T@D + mu*lambda_d**2*np.eye(num_neurons))
    # - Start with zero weights 
    weights_slow = np.zeros((num_neurons,num_neurons))

    # - Pull out thresholds
    v_thresh = (nu * lambda_d + mu * lambda_d**2 + np.sum(abs(D.T), -1, keepdims = True)**2) / 2
    # v_reset = v_thresh - np.reshape(np.diag(weights_fast), (-1, 1))
    # v_rest = v_reset
    # - Fill the diagonal with zeros
    np.fill_diagonal(weights_fast, 0)

    # - Calculate weight matrices for realistic neuron settings
    v_thresh_target = 1.0*np.ones((num_neurons,)) # - V_thresh
    v_rest_target = 0.5*np.ones((num_neurons,)) # - V_rest = b

    b = v_rest_target
    a = v_thresh_target - b

    # - Feedforward weights: Divide each column i by the i-th threshold value and multiply by i-th value of a
    D_realistic = a*np.divide(D, v_thresh.ravel())
    weights_in_realistic = D_realistic
    weights_out_realistic = D_realistic.T
    weights_fast_realistic = a*np.divide(weights_fast.T, v_thresh.ravel()).T # - Divide each row

    # - Reset is given by v_reset_target = b - a
    v_reset_target = b - a
    noise_std_realistic = 0.00


    net = NetworkADS.SpecifyNetwork(N=num_neurons,
                                    Nc=Nc,
                                    Nb=num_neurons,
                                    weights_in=weights_in_realistic * tau_mem,
                                    weights_out= weights_out_realistic / 2,
                                    weights_fast= - weights_fast_realistic * tau_mem / tau_syn_fast,
                                    weights_slow = weights_slow,
                                    eta=0.001,
                                    k=0,
                                    noise_std=noise_std_realistic,
                                    dt=dt,
                                    v_thresh=v_thresh_target,
                                    v_reset=v_reset_target,
                                    v_rest=v_rest_target,
                                    tau_mem=tau_mem,
                                    tau_syn_r_fast=tau_syn_fast,
                                    tau_syn_r_slow=tau_slow,
                                    tau_syn_r_out=tau_out,
                                    discretize=-1,
                                    discretize_dynapse=False,
                                    record=True)


    # - Create TSContinuous and plot
    ts_input = TSContinuous(np.arange(0,duration,duration*dt), inp)
    # ts_input.plot(); plt.show()

    # - Evolve
    fast_weights = net.lyrRes.weights_fast
    net.lyrRes.ts_target = ts_input
    val_sim = net.evolve(ts_input=ts_input, verbose=True); net.reset_all()
    ts_out_val = val_sim["output_layer"]
    v = net.lyrRes._last_evolve["v"]
    vt = net.lyrRes._last_evolve["vt"]

    # - Evolve without the fast recurrent weights
    net.lyrRes.weights_fast *= 0
    val_sim_no_fast = net.evolve(ts_input=ts_input, verbose=True); net.reset_all()
    net.lyrRes.weights_fast = fast_weights
    ts_out_val_no_fast = val_sim_no_fast["output_layer"]
    v_no_fast = net.lyrRes._last_evolve["v"]
    vt_no_fast = net.lyrRes._last_evolve["vt"]

    # - Compute scaling weights, solve least squares problem: arg min A |Y - XA|_2^2
    A_no_fast = np.linalg.pinv(ts_out_val_no_fast.samples.T @ ts_out_val_no_fast.samples) @ ts_out_val_no_fast.samples.T @ ts_input.samples 
    A_fast = np.linalg.pinv(ts_out_val.samples.T @ ts_out_val.samples) @ ts_out_val.samples.T @ ts_input.samples
    x_recon_no_fast =  ts_out_val_no_fast.samples @ A_no_fast
    x_recon_fast = ts_out_val.samples @ A_fast

    error_no_fast = np.sum(np.var(ts_input.samples-x_recon_no_fast, axis=0, ddof=1)) / (np.sum(np.var(ts_input.samples, axis=0, ddof=1)))
    error_fast = np.sum(np.var(ts_input.samples-x_recon_fast, axis=0, ddof=1)) / (np.sum(np.var(ts_input.samples, axis=0, ddof=1)))
    print("Error with fast conn. is",error_fast,"Error without fast connections is",error_no_fast)

    stagger_v = np.ones(v.shape)
    stagger_v_no_fast = np.ones(v_no_fast.shape)
    for idx in range(num_neurons):
        stagger_v[idx,:] += idx
        stagger_v_no_fast[idx,:] += idx

    v_staggered = stagger_v + v
    v_staggered_no_fast = stagger_v_no_fast + v_no_fast

    # - Plot: Smoothed target from net_IO, output from autoencoder, spikes from autoencoder
    t = np.linspace(0,duration,ts_input.samples.shape[0])
    fig = plt.figure(figsize=(20,10),constrained_layout=True)
    gs = fig.add_gridspec(4, 2)
    plot_num = 10
    ax0 = fig.add_subplot(gs[0,:])
    ts_input.plot()
    ax1 = fig.add_subplot(gs[1,0])
    ax1.set_title(r"With fast recurrent weights")
    ax1.plot(t,x_recon_fast)
    ax2 = fig.add_subplot(gs[2,0])
    val_sim["lyrRes"].plot()
    ax2.set_ylim([-0.5,num_neurons-0.5])
    ax3 = fig.add_subplot(gs[3,0])
    ax3.plot(vt, v_staggered[:plot_num,:].T)

    ax5 = fig.add_subplot(gs[1,1])
    ax5.set_title(r"Without fast recurrent weights")
    ax5.plot(t,x_recon_no_fast)
    ax6 = fig.add_subplot(gs[2,1])
    val_sim_no_fast["lyrRes"].plot()
    ax6.set_ylim([-0.5,num_neurons-0.5])
    ax7 = fig.add_subplot(gs[3,1])
    ax7.plot(vt_no_fast, v_staggered_no_fast[:plot_num,:].T)

    plt.tight_layout()
    plt.show()

    inp /= amplitude
