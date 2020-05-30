import numpy as np 
import matplotlib
matplotlib.rc('font', family='Times New Roman')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import json
from Utils import filter_1d

PLOT_FAST_NO_FAST_ROBUSTNESS = False
PLOT_TEMPORAL_XOR = False
PLOT_TRAINING_NUM = False
PLOT_ROBUSTNESS = False
PLOT_HEY_SNIPS = False
PLOT_FAST_NO_FAST_EVOLUTION = False
#### General format: Time x Features (e.g. 5000 x 128)
duration = 5.0
N = 1024

if(PLOT_FAST_NO_FAST_ROBUSTNESS):
    with open("/home/julian/Documents/dm-network-rockpool/Resources/hey-snips/orig_error_fast.npy", "rb") as f:
        orig_error_fast = np.load(f)
    with open("/home/julian/Documents/dm-network-rockpool/Resources/hey-snips/orig_error_no_fast.npy", "rb") as f:
        orig_error_no_fast = np.load(f)
    with open("/home/julian/Documents/dm-network-rockpool/Resources/hey-snips/error_fast.npy", "rb") as f:
        error_fast = np.load(f)
    with open("/home/julian/Documents/dm-network-rockpool/Resources/hey-snips/error_no_fast.npy", "rb") as f:
        error_no_fast = np.load(f)

    # Plot over the course of training
    fig = plt.figure(figsize=(6,1.57))
    alpha = 0.9
    plt.plot(filter_1d(error_fast, alpha=alpha), color="C1", label=r"Error clamped with $\mathbf{\Omega^f}$")
    plt.plot(filter_1d(error_no_fast, alpha=alpha), color="C1", label=r"Error clamped without $\mathbf{\Omega^f}$",linestyle="--")
    plt.plot(filter_1d(orig_error_fast, alpha=alpha), color="C7", label=r"Error original with $\mathbf{\Omega^f}$")
    plt.plot(filter_1d(orig_error_no_fast, alpha=alpha), color="C7", label=r"Error original without $\mathbf{\Omega^f}$", linestyle="--")
    plt.legend(frameon=False, loc=3, prop={'size': 4}, bbox_to_anchor=(0.0,0.0))
    plt.ylim([0.0,0.7])
    plt.ylabel(r"Reconstruction error")
    plt.xlabel(r"Testing iteration")
    plt.tight_layout()
    plt.savefig("/home/julian/Documents/dm-network-rockpool/Latex/figures/figure10.png", dpi=1200)
    plt.show()

elif(PLOT_FAST_NO_FAST_EVOLUTION):
    with open("Resources/hey-snips/node_7_test_acc0.8841158841158842threshold0.7eta0.0001val_acc0.9233870967741935tau_slow0.07tau_out0.07num_neurons768num_dist_weights-1_training_evolution.json", "rb") as f:
        training_evolution_no_fast = json.load(f)
    with open("Resources/hey-snips/node_8_test_acc0.8431568431568431threshold0.7eta0.0001val_acc0.8951612903225806tau_slow0.07tau_out0.07num_neurons768num_dist_weights-1_training_evolution.json", "rb") as f:
        training_evolution_fast = json.load(f)

    fig = plt.figure(figsize=(6,2.3),constrained_layout=True)
    gs = fig.add_gridspec(2, 1)

    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title(r"Training accuracy")
    ax1.set_ylim([0.0,1.0])
    ax2 = fig.add_subplot(gs[1,0])
    ax2.set_title(r"Reconstruction error")
    ax2.set_ylim([0.0,1.0])
    ax2.set_xlabel(r"Training iteration")

    ax1.plot(training_evolution_no_fast["training_acc"], color="C0", linestyle="--", label=r"No $\mathbf{\Omega^f}$")
    ax1.plot(training_evolution_fast["training_acc"], color="r", label=r"With $\mathbf{\Omega^f}$")
    ax1.legend(frameon=False, loc=3, prop={'size': 4})

    ax2.plot(training_evolution_no_fast["training_recon_acc"], color="C0", linestyle="--")
    ax2.plot(training_evolution_fast["training_recon_acc"], color="r")

    axins = zoomed_inset_axes(ax2, 3, loc=1) # zoom = 6
    axins.axes.get_yaxis().set_visible(False)
    axins.axes.get_xaxis().set_visible(False)
    # axins.axis('off')
    axins.plot(training_evolution_no_fast["training_recon_acc"][3500:3700], color="C0", linestyle="--")
    axins.plot(training_evolution_fast["training_recon_acc"][3500:3700], color="r")

    plt.tight_layout()
    plt.savefig("/home/julian/Documents/dm-network-rockpool/Latex/figures/figure9.png", dpi=1200)
    plt.show()

elif(PLOT_TRAINING_NUM):

    training_evolutions = []
    with open("Resources/CloudModels/node_1_training_evolution.json", "rb") as f:
        training_evolutions.append((json.load(f),512))
    with open("Resources/CloudModels/node_2_training_evolution.json", "rb") as f:
        training_evolutions.append((json.load(f),640))
    with open("Resources/CloudModels/node_3_training_evolution.json", "rb") as f:
        training_evolutions.append((json.load(f),768))
    with open("Resources/CloudModels/node_4_training_evolution.json", "rb") as f:
        training_evolutions.append((json.load(f),896))
    with open("Resources/CloudModels/node_5_training_evolution.json", "rb") as f:
        training_evolutions.append((json.load(f),1024))

    total_num_iter = len(training_evolutions[0][0]['training_acc'])
    step_size = 25
    start_k = 200
    stop_k = step_size
    num_reductions = int((start_k - stop_k) / step_size) + 1
    reduce_after = int(total_num_iter / num_reductions)
    reduction_indices = [idx for idx in range(1,total_num_iter) if (idx % reduce_after) == 0]
    k_of_t = np.zeros(total_num_iter)
    if(total_num_iter > 0):
        k_of_t[0] = start_k
        for t in range(1,total_num_iter):
            if(t in reduction_indices):
                k_of_t[t] = k_of_t[t-1]-step_size
            else:
                k_of_t[t] = k_of_t[t-1]
        f_k = lambda t : np.maximum(0,k_of_t[t])

    fig = plt.figure(figsize=(6,2.36),constrained_layout=True)
    gs = fig.add_gridspec(2, 1)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = ax1.twinx()
    ax2.set_ylabel("k", color="r",rotation='horizontal')
    ax1.set_title("Training accuracy")
    ax1.set_ylim([0.0,1.0])
    ax3 = fig.add_subplot(gs[1,0])
    ax3.set_title("Reconstruction error")
    ax3.set_ylim([0.0,1.0])
    ax4 = ax3.twinx()
    ax4.set_ylabel("k", color="r",rotation='horizontal')
    ax3.set_xlabel("Signal iteration")

    for (te,num_neurons) in training_evolutions:
        print(te.keys())
        ax1.plot(te['training_acc'], label=str(num_neurons))
        ax3.plot(te['training_recon_acc'])
    
    ax2.plot(np.arange(0,total_num_iter),f_k(np.arange(0,total_num_iter)),color="r")
    ax4.plot(np.arange(0,total_num_iter),f_k(np.arange(0,total_num_iter)),color="r")

    ax1.legend(frameon=False, loc=3, prop={'size': 4})
    # ax2.legend()
    # ax3.legend()
    plt.savefig("Latex/figures/figure6.png",dpi=1200)
    plt.show()


elif(PLOT_ROBUSTNESS):
    with open('Resources/Plotting/Robustness/spike_channels_original.npy', 'rb') as f:
        spike_channels_original = np.load(f)
    with open('Resources/Plotting/Robustness/spike_times_original.npy', 'rb') as f:
        spike_times_original = np.load(f)
    with open('Resources/Plotting/Robustness/spike_channels_perturbed.npy', 'rb') as f:
        spike_channels_perturbed = np.load(f)
    with open('Resources/Plotting/Robustness/spike_times_perturbed.npy', 'rb') as f:
        spike_times_perturbed = np.load(f)
    with open('Resources/Plotting/Robustness/spike_channels_mismatch_one.npy', 'rb') as f:
        spike_channels_mismatch_one = np.load(f)
    with open('Resources/Plotting/Robustness/spike_times_mismatch_one.npy', 'rb') as f:
        spike_times_mismatch_one = np.load(f)
    with open('Resources/Plotting/Robustness/spike_channels_mismatch_two.npy', 'rb') as f:
        spike_channels_mismatch_two = np.load(f)
    with open('Resources/Plotting/Robustness/spike_times_mismatch_two.npy', 'rb') as f:
        spike_times_mismatch_two = np.load(f)
    with open('Resources/Plotting/Robustness/target_dynamics.npy', 'rb') as f:
        target_dynamics = np.load(f).T
    with open('Resources/Plotting/Robustness/recon_dynamics_original.npy', 'rb') as f:
        recon_dynamics_original = np.load(f).T
        time_dynamics_original = np.arange(0,duration,duration/recon_dynamics_original.shape[0])
        print("recon_dynamics_original",recon_dynamics_original.shape)
    with open('Resources/Plotting/Robustness/recon_dynamics_perturbed.npy', 'rb') as f:
        recon_dynamics_perturbed = np.load(f).T
        time_dynamics_perturbed = np.arange(0,duration,duration/recon_dynamics_perturbed.shape[0])
        print("recon_dynamics_original",recon_dynamics_original.shape)
    with open('Resources/Plotting/Robustness/recon_dynamics_mismatch_one.npy', 'rb') as f:
        recon_dynamics_mismatch_one = np.load(f).T
        time_dynamics_mismatch_one = np.arange(0,duration,duration/recon_dynamics_mismatch_one.shape[0])
        print("recon_dynamics_mismatch_one",recon_dynamics_mismatch_one.shape)
    with open('Resources/Plotting/Robustness/recon_dynamics_mismatch_two.npy', 'rb') as f:
        recon_dynamics_mismatch_two = np.load(f).T
        time_dynamics_mismatch_two = np.arange(0,duration,duration/recon_dynamics_mismatch_two.shape[0])
        print("recon_dynamics_mismatch_two",recon_dynamics_mismatch_two.shape)
    with open('Resources/Plotting/Robustness/spiking_output_original.npy', 'rb') as f:
        final_out_original = np.load(f)
    with open('Resources/Plotting/Robustness/spiking_output_mismatch_one.npy', 'rb') as f:
        final_out_mismatch_one = np.load(f)
    with open('Resources/Plotting/Robustness/spiking_output_mismatch_two.npy', 'rb') as f:
        final_out_mismatch_two = np.load(f)
    with open('Resources/Plotting/Robustness/spiking_output_perturbed.npy', 'rb') as f:
        final_out_perturbed = np.load(f)
    with open('Resources/Plotting/Robustness/rate_output.npy', 'rb') as f:
        rate_output = np.load(f)

    t_start = 1.0
    t_start_dynamics = t_start
    t_stop = 2.7
    t_start_spikes = 1.3
    t_stop_spikes = 1.8
    t_start_perturbed_final = 1.0
    t_stop_perturbed_final = 2.7
    fig = plt.figure(figsize=(6,5.12),constrained_layout=True)
    gs = fig.add_gridspec(10, 8) # Height ratio is 4 : 4 : 2
    # - First row will be reconstructed dynamics vs. target dynamics
    plot_num = 10
    stagger_target_dyn = np.ones((target_dynamics.shape[0],plot_num))
    for i in range(plot_num):
        stagger_target_dyn[:,i] *= i*0.5
    target_dynamics[:,:plot_num] += stagger_target_dyn
    recon_dynamics_original[:,:plot_num] += stagger_target_dyn
    recon_dynamics_mismatch_one[:,:plot_num] += stagger_target_dyn
    recon_dynamics_mismatch_two[:,:plot_num] += stagger_target_dyn
    recon_dynamics_perturbed[:,:plot_num] += stagger_target_dyn
    colors = [("C%d"%i) for i in range(2,plot_num+2)]
    ax1 = fig.add_subplot(gs[:4,:2])
    ax1.set_title(r"Original")
    l1 = ax1.plot(time_dynamics_original[(time_dynamics_original > t_start_dynamics) & (time_dynamics_original < t_stop)], target_dynamics[(time_dynamics_original > t_start_dynamics) & (time_dynamics_original < t_stop),:plot_num], linestyle="--")
    l2 = ax1.plot(time_dynamics_original[(time_dynamics_original > t_start_dynamics) & (time_dynamics_original < t_stop)], recon_dynamics_original[(time_dynamics_original > t_start_dynamics) & (time_dynamics_original < t_stop),:plot_num])
    for line, color in zip(l1,colors):
        line.set_color(color)
    for line, color in zip(l2,colors):
        line.set_color(color)
    lines = [l1[0],l2[0]]
    ax1.legend(lines, [r"$\mathbf{x}$", r"$\tilde{\mathbf{x}}$"], loc=0, prop={'size': 5})
    leg = ax1.get_legend()
    leg.legendHandles[0].set_color('black')
    leg.legendHandles[1].set_color('black')
    ax1.set_ylabel(r"\textbf{A)}")

    ax2 = fig.add_subplot(gs[:4,2:4])
    ax2.set_title(r"Mismatch 1")
    l1 = ax2.plot(time_dynamics_mismatch_one[(time_dynamics_mismatch_one > t_start_dynamics) & (time_dynamics_mismatch_one < t_stop)], target_dynamics[(time_dynamics_mismatch_one > t_start_dynamics) & (time_dynamics_mismatch_one < t_stop),:plot_num], linestyle="--")
    l2 = ax2.plot(time_dynamics_mismatch_one[(time_dynamics_mismatch_one > t_start_dynamics) & (time_dynamics_mismatch_one < t_stop)], recon_dynamics_mismatch_one[(time_dynamics_mismatch_one > t_start_dynamics) & (time_dynamics_mismatch_one < t_stop),:plot_num])
    for line, color in zip(l1,colors):
        line.set_color(color)
    for line, color in zip(l2,colors):
        line.set_color(color)
    ax2.axes.get_yaxis().set_visible(False)

    ax3 = fig.add_subplot(gs[:4,4:6])
    ax3.set_title(r"Mismatch 2")
    l1 = ax3.plot(time_dynamics_mismatch_two[(time_dynamics_mismatch_two > t_start_dynamics) & (time_dynamics_mismatch_two < t_stop)], target_dynamics[(time_dynamics_mismatch_two > t_start_dynamics) & (time_dynamics_mismatch_two < t_stop),:plot_num], linestyle="--")
    l2 = ax3.plot(time_dynamics_mismatch_two[(time_dynamics_mismatch_two > t_start_dynamics) & (time_dynamics_mismatch_two < t_stop)], recon_dynamics_mismatch_two[(time_dynamics_mismatch_two > t_start_dynamics) & (time_dynamics_mismatch_two < t_stop),:plot_num])
    for line, color in zip(l1,colors):
        line.set_color(color)
    for line, color in zip(l2,colors):
        line.set_color(color)
    ax3.axes.get_yaxis().set_visible(False)

 
    ax4 = fig.add_subplot(gs[4:8,:2])
    ax4.scatter(spike_times_original[(spike_times_original > t_start_spikes) & (spike_times_original < t_stop_spikes)], spike_channels_original[(spike_times_original > t_start_spikes) & (spike_times_original < t_stop_spikes)],color='k')
    ax4.set_ylabel(r"\textbf{B)}")
    ax4.set_xlim([t_start_spikes,t_stop_spikes])

    ax5 = fig.add_subplot(gs[4:8,2:4])
    ax5.scatter(spike_times_mismatch_one[(spike_times_mismatch_one > t_start_spikes) & (spike_times_mismatch_one < t_stop_spikes)], spike_channels_mismatch_one[(spike_times_mismatch_one > t_start_spikes) & (spike_times_mismatch_one < t_stop_spikes)],color='k')
    ax5.set_xlim([t_start_spikes,t_stop_spikes])
    ax5.axes.get_yaxis().set_visible(False)

    ax6 = fig.add_subplot(gs[4:8,4:6])
    ax6.scatter(spike_times_mismatch_two[(spike_times_mismatch_two > t_start_spikes) & (spike_times_mismatch_two < t_stop_spikes)], spike_channels_mismatch_two[(spike_times_mismatch_two > t_start_spikes) & (spike_times_mismatch_two < t_stop_spikes)],color='k')
    ax6.set_xlim([t_start_spikes,t_stop_spikes])
    ax6.axes.get_yaxis().set_visible(False)

    ax7 = fig.add_subplot(gs[8:10,:2])
    ax7.plot(np.arange(0,duration,duration/len(rate_output)), rate_output, color="C2", label=r"$\mathbf{y}_{\textnormal{rate}}$")
    ax7.plot(np.arange(0,duration,duration/len(final_out_original)), final_out_original, color="C4", linestyle="--", label=r"$\mathbf{y}_{\textnormal{spiking}}$")
    ax7.axhline(y=0.7, label=r"Threshold")
    ax7.legend(frameon=False, loc=1, prop={'size': 5})
    ax7.set_ylim([-0.4,1.0])
    ax7.set_ylabel(r"\textbf{C)}")

    ax8 = fig.add_subplot(gs[8:10,2:4])
    ax8.plot(np.arange(0,duration,duration/len(rate_output)), rate_output, color="C2", label=r"$\mathbf{y}_{\textnormal{rate}}$")
    ax8.plot(np.arange(0,duration,duration/len(final_out_mismatch_one)), final_out_mismatch_one, color="C4", linestyle="--", label=r"$\mathbf{y}_{\textnormal{spiking}}$")
    ax8.axhline(y=0.7, label=r"Threshold")
    ax8.set_ylim([-0.4,1.0])
    ax8.axes.get_yaxis().set_visible(False)

    ax9 = fig.add_subplot(gs[8:10,4:6])
    ax9.plot(np.arange(0,duration,duration/len(rate_output)), rate_output, color="C2", label=r"$\mathbf{y}_{\textnormal{rate}}$")
    ax9.plot(np.arange(0,duration,duration/len(final_out_mismatch_two)), final_out_mismatch_two, color="C4", linestyle="--", label=r"$\mathbf{y}_{\textnormal{spiking}}$")
    ax9.axhline(y=0.7, label=r"Threshold")
    ax9.set_ylim([-0.4,1.0])
    ax9.axes.get_yaxis().set_visible(False)

    ax10 = fig.add_subplot(gs[:4,6:])
    ax10.set_title(r"Perturbation")
    l1 = ax10.plot(time_dynamics_perturbed[(time_dynamics_perturbed > t_start_dynamics) & (time_dynamics_perturbed < t_stop)], target_dynamics[(time_dynamics_perturbed > t_start_dynamics) & (time_dynamics_perturbed < t_stop),:plot_num], linestyle="--")
    l2 = ax10.plot(time_dynamics_perturbed[(time_dynamics_perturbed > t_start_dynamics) & (time_dynamics_perturbed < t_stop)], recon_dynamics_perturbed[(time_dynamics_perturbed > t_start_dynamics) & (time_dynamics_perturbed < t_stop),:plot_num])
    for line, color in zip(l1,colors):
        line.set_color(color)
    for line, color in zip(l2,colors):
        line.set_color(color)
    ax10.axvline(x=1.4, color='r')
    ax10.axvline(x=1.7, color='r')
    ax10.axes.get_yaxis().set_visible(False)

    ax11 = fig.add_subplot(gs[4:8,6:])
    ax11.scatter(spike_times_perturbed[(spike_times_perturbed > t_start_spikes) & (spike_times_perturbed < t_stop_spikes)], spike_channels_perturbed[(spike_times_perturbed > t_start_spikes) & (spike_times_perturbed < t_stop_spikes)],color='k')
    ax11.set_xlim([t_start_spikes,t_stop_spikes])
    ax11.axvline(x=1.4, color='r')
    ax11.axvline(x=1.7, color='r')
    ax11.axes.get_yaxis().set_visible(False)

    ax12 = fig.add_subplot(gs[8:,6])
    tmp_times = np.arange(0,duration,duration/len(rate_output))
    ax12.plot(tmp_times[(tmp_times > t_start_perturbed_final) & (tmp_times < t_stop_perturbed_final)], rate_output[(tmp_times > t_start_perturbed_final) & (tmp_times < t_stop_perturbed_final)], color="C2", label=r"$\mathbf{y}_{\textnormal{rate}}$")
    ax12.plot(tmp_times[(tmp_times > t_start_perturbed_final) & (tmp_times < t_stop_perturbed_final)], final_out_perturbed[(tmp_times > t_start_perturbed_final) & (tmp_times < t_stop_perturbed_final)], color="C4", linestyle="--", label=r"$\mathbf{y}_{\textnormal{spiking}}$")
    ax12.axhline(y=0.7, label=r"Threshold")
    ax12.set_ylim([-0.4,1.0])
    ax12.axvline(x=1.4, color='r')
    ax12.axvline(x=1.7, color='r')
    ax12.axes.get_yaxis().set_visible(False)
    ax12.axes.get_xaxis().set_visible(False)

    # - Compute MSE for perturbed sample
    mse = np.sum((target_dynamics-recon_dynamics_perturbed)**2,axis=1)
    mse_original = np.sum((target_dynamics-recon_dynamics_original)**2,axis=1)
    ax13 = fig.add_subplot(gs[8:,7])
    t_mse = np.arange(0,duration,duration/len(mse))
    l1 = ax13.plot(t_mse[(t_mse > t_start_spikes) & (t_mse < t_stop_spikes)], mse[(t_mse > t_start_spikes) & (t_mse < t_stop_spikes)], color="C2")
    l2 = ax13.plot(t_mse[(t_mse > t_start_spikes) & (t_mse < t_stop_spikes)], mse_original[(t_mse > t_start_spikes) & (t_mse < t_stop_spikes)], color="C4",linestyle="--")
    lines = [l1[0],l2[0]]
    ax13.legend(lines, [r"Perturbed", r"Original"], loc=0, frameon=False, prop={'size': 3})
    leg = ax13.get_legend()
    leg.legendHandles[0].set_color('black')
    leg.legendHandles[1].set_color('black')
    ax13.set_title(r"MSE")
    for tick in ax13.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
    ax13.axvline(x=1.4, color='r')
    ax13.axvline(x=1.7, color='r')
    ax13.axes.get_yaxis().set_visible(False)

    plt.savefig("Latex/figures/figure4.png", dpi=1200)
    plt.show()

elif(PLOT_HEY_SNIPS):
    # - load voltages, rate dynamics, recon dynamics, rate output, spiking output, fast and slow matrix, raw input, filtered input
    with open('Resources/Plotting/General/v.npy', 'rb') as f:
        v = np.load(f).T
        print("V",v.shape)
    with open('Resources/Plotting/General/times.npy', 'rb') as f:
        times = np.load(f)
        print("Times",times.shape)
    with open('Resources/Plotting/General/target_dynamics.npy', 'rb') as f:
        target_val = np.load(f).T
        print("target_val",target_val.shape)
    with open('Resources/Plotting/General/recon_dynamics.npy', 'rb') as f:
        out_val = np.load(f).T
        time_dynamics = np.arange(0,duration,duration/out_val.shape[0])
        print("out_val",out_val.shape)
    with open('Resources/Plotting/General/rate_output.npy', 'rb') as f:
        rate_output = np.load(f)
        print("rate_output",rate_output.shape)
    with open('Resources/Plotting/General/spiking_output.npy', 'rb') as f:
        final_out = np.load(f)
        print("final_out",final_out.shape)
    with open('Resources/Plotting/General/omega_f.npy', 'rb') as f:
        omega_f = np.load(f)
        print("omega_f",omega_f.shape)
    with open('Resources/Plotting/General/omega_s.npy', 'rb') as f:
        omega_s = np.load(f)
        print("omega_s",omega_s.shape)
    with open('Resources/Plotting/General/audio_raw.npy', 'rb') as f:
        audio_raw = np.load(f)
        times_audio_raw = np.arange(0,duration,duration/len(audio_raw))
        print('audio_raw',audio_raw.shape)
    # - Create time base
    with open('Resources/Plotting/General/filtered_audio.npy', 'rb') as f:
        filtered = np.load(f)
        plot_num = 16
        stagger_filtered = np.ones((filtered.shape[0],plot_num))
        for i in range(plot_num):
            stagger_filtered[:,i] *= i*0.1
        filtered += stagger_filtered
        filtered_times = np.arange(0,duration,duration/filtered.shape[0])
        print("filtered",filtered.shape)
    with open('Resources/Plotting/General/rate_output_false.npy', 'rb') as f:
        rate_output_false = np.load(f)
        print("rate_out_false",rate_output_false.shape)
    with open('Resources/Plotting/General/spiking_output_false.npy', 'rb') as f:
        final_out_false = np.load(f)
        print("final_out_false",final_out_false.shape)
    with open('Resources/Plotting/General/spike_channels.npy', 'rb') as f:
        spike_channels = np.load(f)
    with open('Resources/Plotting/General/spike_times.npy', 'rb') as f:
        spike_times = np.load(f)

    t_start = 1.1
    t_stop = 3.1

    fig = plt.figure(figsize=(6,4.3),constrained_layout=True)
    gs = fig.add_gridspec(12, 2)
    # - Left side
    ax1 = fig.add_subplot(gs[:2,0])
    ax1.set_title(r"\textbf{A)} Raw audio")
    ax1.plot(times_audio_raw[(times_audio_raw > t_start) & (times_audio_raw < t_stop)], audio_raw[(times_audio_raw > t_start) & (times_audio_raw < t_stop)], color="k", linewidth=0.05)
    ax1.axes.get_yaxis().set_visible(False)
    ax2 = fig.add_subplot(gs[2:6,0])
    ax2.set_title(r"\textbf{B)} Filtered input")
    ax2.plot(filtered_times[(filtered_times > t_start) & (filtered_times < t_stop)], filtered[(filtered_times > t_start) & (filtered_times < t_stop),:], color="k")
    ax2.axes.get_yaxis().set_visible(False)
    ax3 = fig.add_subplot(gs[6:9,0])
    ax3.set_title(r"\textbf{C)} Classification of true sample")
    ax3.plot(np.arange(0,duration,duration/len(rate_output)), rate_output, color="C2", label=r"$\mathbf{y}_{\textnormal{rate}}$")
    ax3.plot(np.arange(0,duration,duration/len(final_out)), final_out, color="C4", linestyle="--", label=r"$\mathbf{y}_{\textnormal{spiking}}$")
    ax3.axhline(y=0.7, label=r"Threshold")
    ax3.legend(frameon=False, loc=1, prop={'size': 5})
    ax3.set_ylim([-0.4,1.0])
    ax4 = fig.add_subplot(gs[9:,0])
    ax4.set_title(r"\textbf{D)} Classification of false sample")
    ax4.plot(np.arange(0,duration,duration/len(rate_output_false)), rate_output_false, color="C2", linestyle="--", label=r"$\mathbf{y}_{\textnormal{rate}}$")
    ax4.plot(np.arange(0,duration,duration/len(final_out_false)), final_out_false, color="C4", label=r"$\mathbf{y}_{\textnormal{spiking}}$")
    ax4.axhline(y=0.7, label=r"Threshold")
    ax4.legend(frameon=False, loc=1, prop={'size': 5})
    ax4.set_ylim([-0.4,1.0])
    # ax5 = fig.add_subplot(gs[10:,0])
    # ax5.set_title(r"\textbf{E)} $\Omega_f$")
    # ax5.axis("off")
    # cax = ax5.matshow(omega_f, cmap="RdBu")
    # fig.colorbar(cax)

    # - Right side
    ax6 = fig.add_subplot(gs[:7,1])
    plot_num = 10
    stagger_target_dyn = np.ones((target_val.shape[0],plot_num))
    for i in range(plot_num):
        stagger_target_dyn[:,i] *= i*0.5
    target_val[:,:plot_num] += stagger_target_dyn
    out_val[:,:plot_num] += stagger_target_dyn
    colors = [("C%d"%i) for i in range(2,plot_num+2)]
    t_start_dynamics = 1.2
    l1 = ax6.plot(time_dynamics[(time_dynamics > t_start_dynamics) & (time_dynamics < t_stop)], target_val[(time_dynamics > t_start_dynamics) & (time_dynamics < t_stop),:plot_num], linestyle="--")
    l2 = ax6.plot(time_dynamics[(time_dynamics > t_start_dynamics) & (time_dynamics < t_stop)], out_val[(time_dynamics > t_start_dynamics) & (time_dynamics < t_stop),:plot_num])
    for line, color in zip(l1,colors):
        line.set_color(color)
    for line, color in zip(l2,colors):
        line.set_color(color)
    lines = [l1[0],l2[0]]
    ax6.legend(lines, [r"$\mathbf{x}$", r"$\tilde{\mathbf{x}}$"], loc=0, prop={'size': 5})
    leg = ax6.get_legend()
    leg.legendHandles[0].set_color('black')
    leg.legendHandles[1].set_color('black')
    ax6.set_title(r"\textbf{E)} Reconstructed vs. target dynamics")
    ax6.axes.get_yaxis().set_visible(False)
    ax7 = fig.add_subplot(gs[7:,1])
    plot_num = 10
    ax7.set_title(r"\textbf{F)} Population spike trains")
    t_start_spikes = 1.2
    t_stop_spikes = 1.7
    ax7.scatter(spike_times[(spike_times > t_start_spikes) & (spike_times < t_stop_spikes)], spike_channels[(spike_times > t_start_spikes) & (spike_times < t_stop_spikes)], color="k")
    # ax7.plot(times[(times > t_start_dynamics) & (times < t_stop)], v[(times > t_start_dynamics) & (times < t_stop), :plot_num], color="k")

    # ax8 = fig.add_subplot(gs[10:,1])
    # ax8.set_title(r"\textbf{H)} $\Omega_s$")
    # ax8.axis("off")
    # cax2 = ax8.matshow(omega_s, cmap="RdBu")
    # fig.colorbar(cax2)

    plt.savefig("Latex/figures/figure3.png", dpi=1200)
    plt.show()
    plot_num_v = 100
    stagger_v = np.ones((v.shape[0],plot_num_v))
    for i in range(plot_num_v):
        stagger_v[:,i] *= i
    v[:,:plot_num_v] += stagger_v

    plt.plot(times[(times > t_start) & (times < t_stop)], v[(times > t_start) & (times < t_stop),:plot_num_v])
    plt.show()



elif(PLOT_TEMPORAL_XOR):
    with open('Resources/Plotting/TemporalXOR/final_out_0.npy', 'rb') as f:
        final_out_0 = np.load(f)
    with open('Resources/Plotting/TemporalXOR/final_out_1.npy', 'rb') as f:
        final_out_1 = np.load(f)
    with open('Resources/Plotting/TemporalXOR/final_out_2.npy', 'rb') as f:
        final_out_2 = np.load(f)
    with open('Resources/Plotting/TemporalXOR/final_out_3.npy', 'rb') as f:
        final_out_3 = np.load(f)

    with open('Resources/Plotting/TemporalXOR/input_0.npy', 'rb') as f:
        input_0 = np.load(f)
    with open('Resources/Plotting/TemporalXOR/input_1.npy', 'rb') as f:
        input_1 = np.load(f)
    with open('Resources/Plotting/TemporalXOR/input_2.npy', 'rb') as f:
        input_2 = np.load(f)
    with open('Resources/Plotting/TemporalXOR/input_3.npy', 'rb') as f:
        input_3 = np.load(f)

    with open('Resources/Plotting/TemporalXOR/target_0.npy', 'rb') as f:
        target_0 = np.load(f)
    with open('Resources/Plotting/TemporalXOR/target_1.npy', 'rb') as f:
        target_1 = np.load(f)
    with open('Resources/Plotting/TemporalXOR/target_2.npy', 'rb') as f:
        target_2 = np.load(f)
    with open('Resources/Plotting/TemporalXOR/target_3.npy', 'rb') as f:
        target_3 = np.load(f)

    with open('Resources/Plotting/TemporalXOR/spike_channels.npy', 'rb') as f:
        spike_channels = np.load(f)
    with open('Resources/Plotting/TemporalXOR/spike_times.npy', 'rb') as f:
        spike_times = np.load(f)

    with open('Resources/Plotting/TemporalXOR/reconstructed_dynamics.npy', 'rb') as f:
        reconstructed_dynamics = np.load(f)
        reconstructed_dynamics = reconstructed_dynamics.T
    with open('Resources/Plotting/TemporalXOR/target_dynamics.npy', 'rb') as f:
        target_dynamics = np.load(f)
    plot_num_dyn = 6
    stagger_dyn = np.ones((target_dynamics.shape[0],plot_num_dyn))
    for i in range(plot_num_dyn):
        stagger_dyn[:,i] *= i
    target_dynamics[:,:plot_num_dyn] += stagger_dyn
    reconstructed_dynamics[:,:plot_num_dyn] += stagger_dyn
    
    with open('Resources/Plotting/TemporalXOR/v.npy', 'rb') as f:
        v = np.load(f)
        v = v.T
    with open('Resources/Plotting/TemporalXOR/vt.npy', 'rb') as f:
        vt = np.load(f)
    plot_num_v = 6
    stagger_v = np.ones((v.shape[0],plot_num_v))
    for i in range(plot_num_v):
        stagger_v[:,i] *= i
    v[:,:plot_num_v] += stagger_v
    
    fig = plt.figure(figsize=(6,4.3),constrained_layout=True)
    gs = fig.add_gridspec(6, 2)

    time_base = np.arange(0,1.0,0.001)
    colors_dyn = [("C%d"%i) for i in range(2,plot_num_dyn+2)]

    ax0 = fig.add_subplot(gs[:2,0])
    l1 = ax0.plot(time_base, target_dynamics[:,:plot_num_dyn], linestyle="--")
    l2 = ax0.plot(time_base, reconstructed_dynamics[:,:plot_num_dyn])
    for line, color in zip(l1,colors_dyn):
        line.set_color(color)
    for line, color in zip(l2,colors_dyn):
        line.set_color(color)
    lines = [l1[0],l2[0]]
    ax0.legend(lines, [r"$\mathbf{x}$", r"$\tilde{\mathbf{x}}$"], loc=0, prop={'size': 5})
    leg = ax0.get_legend()
    leg.legendHandles[0].set_color('black')
    leg.legendHandles[1].set_color('black')
    ax0.set_title(r"\textbf{A)} Reconstructed vs. target dynamics")
    ax0.axes.get_yaxis().set_visible(False)

    ax1 = fig.add_subplot(gs[2:4,0])
    ax1.plot(vt, v[:,:plot_num_v], color="k")
    ax1.set_title(r"\textbf{B)} Membrane potentials")
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)

    ax2 = fig.add_subplot(gs[0,1])
    ax2.plot(time_base, input_0, color="k", label=r"c")
    ax2.plot(time_base, target_0, color="g", linestyle="--", label=r"$y_{\textnormal{target}}$")
    ax2.plot(time_base, final_out_0, color="C4", label=r"$y_{\textnormal{spiking}}$")
    ax2.set_title(r"\textbf{C)} Input \& Target vs. network output")
    ax2.legend(frameon=True, loc=1, prop={'size': 5})

    ax3 = fig.add_subplot(gs[1,1])
    ax3.plot(time_base, input_1, color="k", label=r"c")
    ax3.plot(time_base, target_1, color="g", linestyle="--")
    ax3.plot(time_base, final_out_1, color="C4")
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)

    ax4 = fig.add_subplot(gs[2,1])
    ax4.plot(time_base, input_2, color="k", label=r"c")
    ax4.plot(time_base, target_2, color="g", linestyle="--")
    ax4.plot(time_base, final_out_2, color="C4")
    ax4.axes.get_xaxis().set_visible(False)
    ax4.axes.get_yaxis().set_visible(False)

    ax5 = fig.add_subplot(gs[3,1])
    ax5.plot(time_base, input_3, color="k", label=r"c")
    ax5.plot(time_base, target_3, color="g", linestyle="--")
    ax5.plot(time_base, final_out_3, color="C4")
    ax5.axes.get_xaxis().set_visible(False)
    ax5.axes.get_yaxis().set_visible(False)

    ax6 = fig.add_subplot(gs[4:6,:])
    ax6.scatter(spike_times, spike_channels, color="k")
    ax6.axes.get_xaxis().set_visible(False)

    plt.savefig("/home/julian/Documents/dm-network-rockpool/Latex/figures/figure2.png", dpi=1200)
    plt.show()