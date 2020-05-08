import warnings
warnings.filterwarnings('ignore')
from Utils import filter_1d
import time
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
from rockpool.timeseries import TSContinuous, TSEvent
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
from tqdm import tqdm
import argparse

# - Change current directory to directory where this file is located
absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)


class HeySnipsNetworkADS(BaseModel):
    def __init__(self,
                 labels,
                 validation_step,
                 num_neurons,
                 tau_slow,
                 tau_out,
                 num_val,
                 num_test,
                 num_epochs,
                 threshold,
                 eta,
                 fs=16000.,
                 verbose=0,
                 name="Snips ADS",
                 version="1.0"):
        
        super(HeySnipsNetworkADS, self).__init__(name,version)

        self.verbose = verbose
        self.fs = fs
        self.dt = 0.001

        self.num_val = num_val
        self.validation_step = validation_step
        self.num_test = num_test

        self.num_epochs = num_epochs
        self.threshold = threshold

        self.N_filter = 51

        self.num_rate_neurons = 128 
        self.num_targets = len(labels)


        with open("Resources/hey-snips/rate_heysnips_tanh_0_16.model", "r") as f:
            config = json.load(f)

        self.w_in = np.array(config['w_in'])
        w_rec = np.array(config['w_recurrent'])
        self.w_out = np.array(config['w_out'])
        self.bias = config['bias']
        tau = config['tau']

        self.rate_layer = RecRateEulerJax_IO(w_in=self.w_in,
                                             w_recurrent=w_rec,
                                             w_out=self.w_out,
                                             tau=tau,
                                             bias=self.bias,
                                             activation_func=H_tanh,
                                             dt=self.dt,
                                             noise_std=0.0,
                                             name="hidden")
        self.num_channels = self.w_in.shape[0]

        self.lyr_filt = ButterMelFilter(fs=fs,
                                num_filters=self.num_channels,
                                cutoff_fs=400.,
                                filter_width=2.,
                                num_workers=4,
                                name='filter')


        # - Create NetworkADS
        model_path_ads_net = "Resources/hey-snips/test_acc0.8471760797342193threshold0.7eta2.2077035344734782e-06val_acc0.86tau_slow0.07tau_out0.07num_neurons1024.json"

        if(os.path.exists(model_path_ads_net)):
            self.net = NetworkADS.load(model_path_ads_net)
            self.net_original = NetworkADS.load(model_path_ads_net)

            N = self.net.lyrRes.size
            Nc = self.net.lyrRes.out_size

            # - Test robustness: Use distribution for tau_fast, tau_out, tau_slow and V_thresh
            mean_tau_slow = self.net.lyrRes.tau_syn_r_slow
            mean_tau_fast = self.net.lyrRes.tau_syn_r_fast
            mean_tau_out = self.net.lyrRes.tau_syn_r_out
            mean_tau_mem = self.net.lyrRes.tau_mem

            self.net.lyrRes.tau_syn_r_slow = np.random.randn(N)*0.2*mean_tau_slow + mean_tau_slow
            self.net.lyrRes.tau_syn_r_fast = np.random.randn(N)*0.2*mean_tau_fast + mean_tau_fast
            self.net.lyrRes.tau_syn_r_out = np.random.randn(Nc)*0.2*mean_tau_out + mean_tau_out
            self.net.lyrRes.tau_mem = np.random.randn(N)*0.2*mean_tau_mem + mean_tau_mem
            # self.net.lyrRes.v_thresh = np.random.randn(N)*0.2*np.mean(self.net.lyrRes.v_thresh) + np.mean(self.net.lyrRes.v_thresh)

            # self.net.lyrRes.weights_fast = np.zeros((N,N))
            # self.net.lyrRes.weights = np.zeros((N,N))
            
            # Weights pruning


            # - Plotting
            plt.subplot(511)
            plt.hist(self.net.lyrRes.tau_syn_r_slow); plt.title("Tau slow")
            plt.subplot(512)
            plt.hist(self.net.lyrRes.tau_syn_r_fast); plt.title("Tau fast")
            plt.subplot(513)
            plt.hist(self.net.lyrRes.tau_syn_r_out); plt.title("Tau out")
            plt.subplot(514)
            plt.hist(self.net.lyrRes.tau_mem); plt.title("Tau mem")
            plt.subplot(515)
            plt.plot(self.net.lyrRes.v_thresh); plt.title("V thresh")
            plt.tight_layout(); plt.show()

            self.num_neurons = self.net.lyrRes.weights_fast.shape[0]
            self.tau_slow = self.net.lyrRes.tau_syn_r_slow
            self.tau_out = self.net.lyrRes.tau_syn_r_out
            self.amplitude = 50 / mean_tau_mem

            plt.figure(figsize=(12,5))
            plt.subplot(121)
            plt.matshow(self.net.lyrRes.weights_slow, cmap="RdBu", fignum=False)
            plt.colorbar()
            plt.subplot(122)
            plt.hist(self.net.lyrRes.weights_slow.ravel(), bins=100)
            plt.title("Weight distribution of slow weights")
            plt.show()

            print("Loaded pretrained network from %s" % model_path_ads_net)
        else:
            print("Model can't be found.")
            raise(Exception)

    def save(self, fn):
        return

    def get_data(self, audio_raw):
        # - Create time base
        times_filt = np.arange(0, len(audio_raw) / self.fs, 1/self.fs)
        # - Create TSContinuos for rate_layer input
        ts_audio_raw = TSContinuous(times_filt, audio_raw)
        # - Get the butterworth input
        ts_filt = self.lyr_filt.evolve(ts_audio_raw)
        self.lyr_filt.reset_all()
        # - Pass through the rate network
        ts_rate_out = self.rate_layer.evolve(ts_filt)
        self.rate_layer.reset_all()
        # - Get the target dynamics
        ts_rate_net_target_dynamics = self.rate_layer.res_acts_last_evolution
        # - Get the input into the spiking network
        ts_spiking_in = TSContinuous(self.rate_layer.res_inputs_last_evolution.times,self.amplitude*self.rate_layer.res_inputs_last_evolution.samples)
        return (ts_spiking_in, ts_rate_net_target_dynamics, ts_rate_out)


    def perform_validation_set(self, data_loader, fn_metrics):
        return


    def train(self, data_loader, fn_metrics):

        for _ in range(self.num_epochs):
            epoch_loss = 0
            yield {"train_loss": epoch_loss}


    def test(self, data_loader, fn_metrics):

        correct = 0
        correct_original = 0
        correct_rate = 0
        count = 0
        for batch_id, [batch, test_logger] in enumerate(data_loader.test_set()):

            if batch_id > self.num_test:
                break

            audio_raw = batch[0][0]
            (ts_spiking_in, ts_rate_net_target_dynamics, ts_rate_out) = self.get_data(audio_raw=audio_raw)
            self.net.lyrRes.ts_target = ts_rate_net_target_dynamics
            self.net_original.lyrRes.ts_target = ts_rate_net_target_dynamics
            val_sim = self.net.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1))
            val_sim_original = self.net_original.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1))

            out_val = val_sim["output_layer"].samples.T
            out_val_original = val_sim_original["output_layer"].samples.T

            self.net.reset_all()
            self.net_original.reset_all()
            
            # - Compute the final classification output
            final_out = out_val.T @ self.w_out
            final_out = filter_1d(final_out, alpha=0.95)

            # - Compute the final classification output
            final_out_original = out_val_original.T @ self.w_out
            final_out_original = filter_1d(final_out_original, alpha=0.95)

            # check for threshold crossing
            if ((final_out > self.threshold).any()):
                predicted_label = 1
            else:
                predicted_label = 0
            if((ts_rate_out.samples > 0.7).any()):
                predicted_label_rate = 1
            else:
                predicted_label_rate = 0
            if ((final_out_original > self.threshold).any()):
                predicted_label_original = 1
            else:
                predicted_label_original = 0

            tgt_label = batch[0][1]
            if(predicted_label == tgt_label):
                correct += 1
            if(predicted_label_rate == tgt_label):
                correct_rate += 1
            if(predicted_label_original == tgt_label):
                correct_original += 1
            count += 1

            # if(self.verbose > 0):
            #     target_val = ts_rate_net_target_dynamics.samples.T
            #     plot_num = 10
            #     stagger = np.ones((plot_num, out_val.shape[1]))
            #     for i in range(plot_num):
            #         stagger[i,:] *= i*0.5

            #     colors = [("C%d"%i) for i in range(2,plot_num+2)]
            #     fig = plt.figure(figsize=(20,6))
            #     ax0 = fig.add_subplot(411)
            #     l1 = ax0.plot(np.linspace(0,out_val.shape[1]*self.dt,out_val.shape[1]), (stagger+out_val[0:plot_num,:]).T)
            #     for line, color in zip(l1,colors):
            #         line.set_color(color)
            #     l2 = ax0.plot(np.linspace(0,target_val.shape[1]*self.dt,target_val.shape[1]), (stagger+target_val[0:plot_num,:]).T, linestyle="--")
            #     for line, color in zip(l2,colors):
            #         line.set_color(color)
            #     ax0.set_title(r"Target vs reconstruction (Mismatch)")
            #     lines = [l1[0],l2[0]]
            #     ax0.legend(lines, ["Reconstruction", "Target"])

            #     ax1 = fig.add_subplot(412)
            #     ts_rate_out.plot()
            #     ax1.set_title("Target output")
            #     ax1.axhline(y=0.7)
            #     ax1.set_ylim([-0.5,1.0])

            #     ax2 = fig.add_subplot(413)
            #     ax2.plot(np.linspace(0,len(final_out)*self.dt,len(final_out)), final_out, label='Mismatch', linestyle='--')
            #     ax2.plot(np.linspace(0,len(final_out_original)*self.dt,len(final_out_original)), final_out_original, label='Original')
            #     ax2.set_title("Spiking output")
            #     ax2.axhline(y=self.threshold)
            #     ax2.set_ylim([-0.5,1.0])
            #     ax2.legend()

            #     ax3 = fig.add_subplot(414)
            #     ax3.plot(np.linspace(0,out_val.shape[1]*self.dt,out_val.shape[1]),np.sum((out_val-target_val)**2, axis=0) / out_val.shape[0], label="Mismatch")
            #     ax3.plot(np.linspace(0,out_val_original.shape[1]*self.dt,out_val_original.shape[1]),np.sum((out_val_original-target_val)**2, axis=0) / out_val_original.shape[0], label="No mismatch")
            #     ax3.legend()
            #     ax3.set_title("MSE over time between spiking network and rate network dynamics")

            #     plt.tight_layout()
            #     # plt.draw()
            #     # plt.waitforbuttonpress(0)
            #     # plt.close(fig)
            #     plt.show()


            target = batch[0][2]
            target_times = np.arange(0, len(target) / self.fs, 1/self.fs)

            plt.clf()
            plt.plot(np.arange(0,len(final_out)*self.dt, self.dt),final_out, label="Mismatch")
            plt.plot(np.arange(0,len(final_out_original)*self.dt, self.dt),final_out_original, label="No mismatch")
            plt.plot(target_times, target, label="Target")
            plt.plot(np.arange(0,len(ts_rate_out.samples)*self.dt, self.dt),ts_rate_out.samples, label="Rate")
            plt.axhline(y=self.threshold)
            plt.ylim([-0.5,1.0])
            plt.legend()
            plt.draw()
            plt.pause(0.001)

            print("--------------------------------")
            print("TESTING batch", batch_id)
            print("True label", tgt_label, "Mismatch", predicted_label, "No mismatch", predicted_label_original, "Rate label", predicted_label_rate)
            print("--------------------------------")

            test_logger.add_predictions(pred_labels=[predicted_label], pred_target_signals=[ts_rate_out.samples])
            fn_metrics('test', test_logger)

        test_acc = correct / count
        test_acc_original = correct_original / count
        test_acc_rate = correct_rate / count
        print("Mismatch test accuracy is %.4f Original test accuracy is %.4f Rate network test accuracy is %.4f" % (test_acc, test_acc_original, test_acc_rate))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--threshold', default=0.7, type=float, help="Threshold for prediction")
    parser.add_argument('--num_test', default=100, type=float, help="Number of test samples")

    args = vars(parser.parse_args())
    verbose = args['verbose']
    threshold = args['threshold']
    num_test = args['num_test']


    batch_size = 1
    percentage_data = 0.02
    # percentage_data = 1.0
    balance_ratio = 1.0
    snr = 10.

    experiment = HeySnipsDEMAND(batch_size=batch_size,
                                percentage=percentage_data,
                                snr=snr,
                                is_tracking=False,
                                one_hot=False)

    num_train_batches = int(np.ceil(experiment.num_train_samples / batch_size))
    num_val_batches = int(np.ceil(experiment.num_val_samples / batch_size))
    num_test_batches = int(np.ceil(experiment.num_test_samples / batch_size))

    model = HeySnipsNetworkADS(labels=experiment._data_loader.used_labels,
                                validation_step=-1,
                                num_neurons=-1,
                                tau_slow=-1,
                                tau_out=-1,
                                num_val=-1,
                                num_test=num_test,
                                num_epochs=0,
                                threshold=threshold,
                                eta=-1,
                                verbose=verbose)

    experiment.set_model(model)
    experiment.set_config({'num_train_batches': num_train_batches,
                           'num_val_batches': num_val_batches,
                           'num_test_batches': num_test_batches,
                           'batch size': batch_size,
                           'percentage data': percentage_data,
                           'snr': snr,
                           'balance_ratio': balance_ratio})
    experiment.start()

    print("experiment done")

    print(f"Accuracy score: {experiment.acc_scores}")

    print("confusion matrix")
    print(experiment.cm)
