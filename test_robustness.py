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
                 mismatch_std,
                 num_bits,
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
        self.mismatch_std = mismatch_std

        self.num_val = num_val
        self.validation_step = validation_step
        self.num_test = num_test

        self.num_epochs = num_epochs
        self.threshold = threshold


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
        model_path_ads_net = "Resources/CloudModels/node_1_test_acc0.8401598401598401threshold0.7eta0.0001val_acc0.86tau_slow0.07tau_out0.07num_neurons1024.json"
        model_path_ads_net_3bit = "Resources/CloudModels/node_1_test_acc0.8401598401598401threshold0.7eta0.0001val_acc0.86tau_slow0.07tau_out0.07num_neurons1024.json"
        model_path_ads_net_4bit = "Resources/CloudModels/node_1_test_acc0.8401598401598401threshold0.7eta0.0001val_acc0.86tau_slow0.07tau_out0.07num_neurons1024.json"

        if(os.path.exists(model_path_ads_net)):
            self.net_mismatch_one = NetworkADS.load(model_path_ads_net)
            self.net_mismatch_two = NetworkADS.load(model_path_ads_net)
            self.net_original = NetworkADS.load(model_path_ads_net)
            self.net_discretized_4_bit = NetworkADS.load(model_path_ads_net_4bit)
            self.net_discretized_3_bit = NetworkADS.load(model_path_ads_net_3bit)

            N = self.net_mismatch_one.lyrRes.size
            Nc = self.net_mismatch_one.lyrRes.out_size

            # - Test robustness: Use distribution for tau_fast, tau_out, tau_slow and V_thresh
            mean_tau_slow = self.net_mismatch_one.lyrRes.tau_syn_r_slow
            mean_tau_fast = self.net_mismatch_one.lyrRes.tau_syn_r_fast
            mean_tau_out = self.net_mismatch_one.lyrRes.tau_syn_r_out
            mean_tau_mem = self.net_mismatch_one.lyrRes.tau_mem

            # Apply first mismatch
            self.net_mismatch_one.lyrRes.tau_syn_r_slow = np.abs(np.random.randn(N)*self.mismatch_std*mean_tau_slow + mean_tau_slow)
            self.net_mismatch_one.lyrRes.tau_syn_r_fast = np.abs(np.random.randn(N)*self.mismatch_std*mean_tau_fast + mean_tau_fast)
            self.net_mismatch_one.lyrRes.tau_syn_r_out = np.abs(np.random.randn(Nc)*self.mismatch_std*mean_tau_out + mean_tau_out)
            self.net_mismatch_one.lyrRes.tau_mem = np.abs(np.random.randn(N)*self.mismatch_std*mean_tau_mem + mean_tau_mem)
            self.net_mismatch_one.lyrRes.v_thresh = np.abs(np.random.randn(N)*self.mismatch_std*np.mean(self.net_mismatch_one.lyrRes.v_thresh) + np.mean(self.net_mismatch_one.lyrRes.v_thresh))
            
            # Apply second mismatch
            self.net_mismatch_two.lyrRes.tau_syn_r_slow = np.abs(np.random.randn(N)*self.mismatch_std*mean_tau_slow + mean_tau_slow)
            self.net_mismatch_two.lyrRes.tau_syn_r_fast = np.abs(np.random.randn(N)*self.mismatch_std*mean_tau_fast + mean_tau_fast)
            self.net_mismatch_two.lyrRes.tau_syn_r_out = np.abs(np.random.randn(Nc)*self.mismatch_std*mean_tau_out + mean_tau_out)
            self.net_mismatch_two.lyrRes.tau_mem = np.abs(np.random.randn(N)*self.mismatch_std*mean_tau_mem + mean_tau_mem)
            self.net_mismatch_two.lyrRes.v_thresh = np.abs(np.random.randn(N)*self.mismatch_std*np.mean(self.net_mismatch_two.lyrRes.v_thresh) + np.mean(self.net_mismatch_two.lyrRes.v_thresh))

            plt.subplot(121)
            plt.hist(self.net_discretized_4_bit.lyrRes.weights_slow.ravel(), bins=2**num_bits)
            plt.subplot(122)
            plt.hist(self.net_original.lyrRes.weights_slow.ravel(), bins=50)
            plt.show()

            # - Plotting
            plt.subplot(511)
            plt.hist(self.net_mismatch_one.lyrRes.tau_syn_r_slow); plt.title("Tau slow")
            plt.subplot(512)
            plt.hist(self.net_mismatch_one.lyrRes.tau_syn_r_fast); plt.title("Tau fast")
            plt.subplot(513)
            plt.hist(self.net_mismatch_one.lyrRes.tau_syn_r_out); plt.title("Tau out")
            plt.subplot(514)
            plt.hist(self.net_mismatch_one.lyrRes.tau_mem); plt.title("Tau mem")
            plt.subplot(515)
            plt.plot(self.net_mismatch_one.lyrRes.v_thresh); plt.title("V thresh")
            plt.tight_layout(); plt.show()

            self.num_neurons = self.net_mismatch_one.lyrRes.weights_fast.shape[0]
            self.tau_slow = self.net_mismatch_one.lyrRes.tau_syn_r_slow
            self.tau_out = self.net_mismatch_one.lyrRes.tau_syn_r_out
            self.amplitude = 50 / mean_tau_mem

            plt.figure(figsize=(12,5))
            plt.subplot(141)
            plt.matshow(self.net_mismatch_one.lyrRes.weights_slow, cmap="RdBu", fignum=False)
            plt.subplot(142)
            plt.matshow(self.net_mismatch_one.lyrRes.weights_fast, cmap="RdBu", fignum=False)
            plt.colorbar()
            plt.subplot(143)
            plt.hist(self.net_mismatch_one.lyrRes.weights_slow.ravel(), bins=100)
            plt.title("Weight distribution of slow weights")
            plt.subplot(144)
            plt.hist(self.net_mismatch_one.lyrRes.weights_fast.ravel(), bins=100)
            plt.title("Weight distribution of fast weights")
            plt.tight_layout()
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

        correct_mismatch_one = 0
        correct_mismatch_two = 0
        correct_perturbed = 0
        correct_original = 0
        correct_discretized_4_bit = 0
        correct_discretized_3_bit = 0
        correct_rate = 0
        count = 0
        already_saved = False
        t_start_suppress = 1.4
        t_stop_suppress = 1.7
        percentage_suppress = 0.2

        for batch_id, [batch, test_logger] in enumerate(data_loader.test_set()):

            if batch_id > self.num_test:
                break

            audio_raw = batch[0][0]
            (ts_spiking_in, ts_rate_net_target_dynamics, ts_rate_out) = self.get_data(audio_raw=audio_raw)
            self.net_mismatch_one.lyrRes.ts_target = ts_rate_net_target_dynamics
            self.net_mismatch_two.lyrRes.ts_target = ts_rate_net_target_dynamics
            self.net_discretized_4_bit.lyrRes.ts_target = ts_rate_net_target_dynamics
            self.net_discretized_3_bit.lyrRes.ts_target = ts_rate_net_target_dynamics
            self.net_original.lyrRes.ts_target = ts_rate_net_target_dynamics
            val_sim_mismatch_one = self.net_mismatch_one.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1))
            val_sim_mismatch_two = self.net_mismatch_two.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1))
            val_sim_discretized_4_bit = self.net_discretized_4_bit.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1))
            val_sim_discretized_3_bit = self.net_discretized_3_bit.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1))
            val_sim_original = self.net_original.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1)); self.net_original.reset_all()
            # - Set the "suppress" fields, they are reset by reset_all()
            self.net_original.lyrRes.t_start_suppress = t_start_suppress
            self.net_original.lyrRes.t_stop_suppress = t_stop_suppress
            self.net_original.lyrRes.percentage_suppress = percentage_suppress
            val_sim_perturbed = self.net_original.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1))

            out_val_mismatch_one = val_sim_mismatch_one["output_layer"].samples.T
            out_val_mismatch_two = val_sim_mismatch_two["output_layer"].samples.T
            out_val_original = val_sim_original["output_layer"].samples.T
            out_val_perturbed = val_sim_perturbed["output_layer"].samples.T
            out_val_discretized_4_bit = val_sim_discretized_4_bit["output_layer"].samples.T
            out_val_discretized_3_bit = val_sim_discretized_3_bit["output_layer"].samples.T

            self.net_mismatch_one.reset_all()
            self.net_mismatch_two.reset_all()
            self.net_original.reset_all()
            self.net_discretized_4_bit.reset_all()
            self.net_discretized_3_bit.reset_all()
            
            # - Compute the final classification output for mismatch one
            final_out_mismatch_one = out_val_mismatch_one.T @ self.w_out
            final_out_mismatch_one = filter_1d(final_out_mismatch_one, alpha=0.95)

            # - Compute the final classification output for mismatch two
            final_out_mismatch_two = out_val_mismatch_two.T @ self.w_out
            final_out_mismatch_two = filter_1d(final_out_mismatch_two, alpha=0.95)

            # - Compute the final classification output of original net
            final_out_original = out_val_original.T @ self.w_out
            final_out_original = filter_1d(final_out_original, alpha=0.95)

            # - Compute the final classification output of perturbed net
            final_out_perturbed = out_val_perturbed.T @ self.w_out
            final_out_perturbed = filter_1d(final_out_perturbed, alpha=0.95)

            # - Compute the final classification output of discretized net
            final_out_discretized_4_bit = out_val_discretized_4_bit.T @ self.w_out
            final_out_discretized_4_bit = filter_1d(final_out_discretized_4_bit, alpha=0.95)

            # - Compute the final classification output of discretized net
            final_out_discretized_3_bit = out_val_discretized_3_bit.T @ self.w_out
            final_out_discretized_3_bit = filter_1d(final_out_discretized_3_bit, alpha=0.95)

            # - Check for threshold crossing of mismatch one
            if ((final_out_mismatch_one > self.threshold).any()):
                predicted_label_mismatch_one = 1
            else:
                predicted_label_mismatch_one = 0
            # - Check for threshold crossing of mismatch two
            if ((final_out_mismatch_two > self.threshold).any()):
                predicted_label_mismatch_two = 1
            else:
                predicted_label_mismatch_two = 0
            # - Check for crossing of rate net
            if((ts_rate_out.samples > 0.7).any()):
                predicted_label_rate = 1
            else:
                predicted_label_rate = 0
            # - Check for crossing of the original net
            if ((final_out_original > self.threshold).any()):
                predicted_label_original = 1
            else:
                predicted_label_original = 0
            # - Check for crossing of the perturbed net
            if ((final_out_perturbed > self.threshold).any()):
                predicted_label_perturbed = 1
            else:
                predicted_label_perturbed = 0
            # - Check for crossing of the discretized net
            if ((final_out_discretized_4_bit > self.threshold).any()):
                predicted_label_discretized_4_bit = 1
            else:
                predicted_label_discretized_4_bit = 0
            # - Check for crossing of the discretized 3bit net
            if ((final_out_discretized_3_bit > self.threshold).any()):
                predicted_label_discretized_3_bit = 1
            else:
                predicted_label_discretized_3_bit = 0

            tgt_label = batch[0][1]
            if(predicted_label_mismatch_one == tgt_label):
                correct_mismatch_one += 1
            if(predicted_label_mismatch_two == tgt_label):
                correct_mismatch_two += 1
            if(predicted_label_rate == tgt_label):
                correct_rate += 1
            if(predicted_label_original == tgt_label):
                correct_original += 1
            if(predicted_label_perturbed == tgt_label):
                correct_perturbed += 1
            if(predicted_label_discretized_4_bit == tgt_label):
                correct_discretized_4_bit += 1
            if(predicted_label_discretized_3_bit == tgt_label):
                correct_discretized_3_bit += 1
            count += 1


            # # - Save a bunch of data for plotting
            # if(tgt_label == 1 and predicted_label_mismatch_one == 1 and predicted_label_mismatch_two == 1 and predicted_label_rate == 1 and predicted_label_original == 1 and not already_saved):
            #     already_saved = True
            #     # - Save rate dynamics, recon dynamics_orig/m1/m2, rate output, spiking output_orig/m1/m2
            #     with open('Resources/Plotting/Robustness/target_dynamics.npy', 'wb') as f:
            #         np.save(f, ts_rate_net_target_dynamics.samples.T)
            #     with open('Resources/Plotting/Robustness/recon_dynamics_original.npy', 'wb') as f:
            #         np.save(f, out_val_original)
            #     with open('Resources/Plotting/Robustness/recon_dynamics_mismatch_one.npy', 'wb') as f:
            #         np.save(f, out_val_mismatch_one)
            #     with open('Resources/Plotting/Robustness/recon_dynamics_mismatch_two.npy', 'wb') as f:
            #         np.save(f, out_val_mismatch_two)
            #     with open('Resources/Plotting/Robustness/recon_dynamics_perturbed.npy', 'wb') as f:
            #         np.save(f, out_val_perturbed)
            #     with open('Resources/Plotting/Robustness/rate_output.npy', 'wb') as f:
            #         np.save(f, ts_rate_out.samples)
            #     with open('Resources/Plotting/Robustness/spiking_output_original.npy', 'wb') as f:
            #         np.save(f, final_out_original)
            #     with open('Resources/Plotting/Robustness/spiking_output_mismatch_one.npy', 'wb') as f:
            #         np.save(f, final_out_mismatch_one)
            #     with open('Resources/Plotting/Robustness/spiking_output_mismatch_two.npy', 'wb') as f:
            #         np.save(f, final_out_mismatch_two)
            #     with open('Resources/Plotting/Robustness/spiking_output_perturbed.npy', 'wb') as f:
            #         np.save(f, final_out_perturbed)
            #     # - Create time base
            #     times_filt = np.arange(0, len(audio_raw) / self.fs, 1/self.fs)
            #     # - Create TSContinuos for rate_layer input
            #     ts_audio_raw = TSContinuous(times_filt, audio_raw)
            #     # - Get the butterworth input
            #     filtered = self.lyr_filt.evolve(ts_audio_raw).samples
            #     self.lyr_filt.reset_all()
            #     with open('Resources/Plotting/Robustness/filtered_audio.npy', 'wb') as f:
            #         np.save(f, filtered)
                
            #     channels_original = val_sim_original["lyrRes"].channels[val_sim_original["lyrRes"].channels >= 0]
            #     times_tmp_original = val_sim_original["lyrRes"].times[val_sim_original["lyrRes"].channels >= 0]
            #     with open('Resources/Plotting/Robustness/spike_channels_original.npy', 'wb') as f:
            #         np.save(f, channels_original)
            #     with open('Resources/Plotting/Robustness/spike_times_original.npy', 'wb') as f:
            #         np.save(f, times_tmp_original)

            #     channels_mismatch_one = val_sim_mismatch_one["lyrRes"].channels[val_sim_mismatch_one["lyrRes"].channels >= 0]
            #     times_tmp_mismatch_one = val_sim_mismatch_one["lyrRes"].times[val_sim_mismatch_one["lyrRes"].channels >= 0]
            #     with open('Resources/Plotting/Robustness/spike_channels_mismatch_one.npy', 'wb') as f:
            #         np.save(f, channels_mismatch_one)
            #     with open('Resources/Plotting/Robustness/spike_times_mismatch_one.npy', 'wb') as f:
            #         np.save(f, times_tmp_mismatch_one)

            #     channels_mismatch_two = val_sim_mismatch_two["lyrRes"].channels[val_sim_mismatch_two["lyrRes"].channels >= 0]
            #     times_tmp_mismatch_two = val_sim_mismatch_two["lyrRes"].times[val_sim_mismatch_two["lyrRes"].channels >= 0]
            #     with open('Resources/Plotting/Robustness/spike_channels_mismatch_two.npy', 'wb') as f:
            #         np.save(f, channels_mismatch_two)
            #     with open('Resources/Plotting/Robustness/spike_times_mismatch_two.npy', 'wb') as f:
            #         np.save(f, times_tmp_mismatch_two)

            #     channels_perturbed = val_sim_perturbed["lyrRes"].channels[val_sim_perturbed["lyrRes"].channels >= 0]
            #     times_tmp_perturbed = val_sim_perturbed["lyrRes"].times[val_sim_perturbed["lyrRes"].channels >= 0]
            #     with open('Resources/Plotting/Robustness/spike_channels_perturbed.npy', 'wb') as f:
            #         np.save(f, channels_perturbed)
            #     with open('Resources/Plotting/Robustness/spike_times_perturbed.npy', 'wb') as f:
            #         np.save(f, times_tmp_perturbed)


            target = batch[0][2]
            target_times = np.arange(0, len(target) / self.fs, 1/self.fs)

            if(self.verbose > 0):
                plt.clf()
                # plt.plot(np.arange(0,len(final_out_mismatch_one)*self.dt, self.dt),final_out_mismatch_one, label="Mismatch 1")
                # plt.plot(np.arange(0,len(final_out_mismatch_two)*self.dt, self.dt),final_out_mismatch_two, label="Mismatch 2")
                plt.plot(np.arange(0,len(final_out_original)*self.dt, self.dt),final_out_original, label="No mismatch")
                # plt.plot(np.arange(0,len(final_out_perturbed)*self.dt, self.dt),final_out_perturbed, label="Perturbed")
                plt.plot(np.arange(0,len(final_out_discretized_4_bit)*self.dt, self.dt),final_out_discretized_4_bit, label="Discretized 4bit")
                plt.plot(np.arange(0,len(final_out_discretized_3_bit)*self.dt, self.dt),final_out_discretized_4_bit, label="Discretized 3bit")
                plt.plot(target_times, target, label="Target")
                plt.plot(np.arange(0,len(ts_rate_out.samples)*self.dt, self.dt),ts_rate_out.samples, label="Rate")
                plt.axhline(y=self.threshold)
                plt.ylim([-0.5,1.0])
                plt.legend()
                plt.draw()
                plt.pause(0.001)

            # print("--------------------------------")
            # print("TESTING batch", batch_id)
            # print("True label", tgt_label, "Mismatch-One", predicted_label_mismatch_one, "Mismatch-Two", predicted_label_mismatch_two, "No mismatch", predicted_label_original, "Discretized 4bit", predicted_label_discretized_4_bit, "Discretized 3bit", predicted_label_discretized_3_bit, "Rate label", predicted_label_rate)
            # print("--------------------------------")

            test_logger.add_predictions(pred_labels=[predicted_label_mismatch_one], pred_target_signals=[ts_rate_out.samples])
            fn_metrics('test', test_logger)

        test_acc_mismatch_one = correct_mismatch_one / count
        test_acc_mismatch_two = correct_mismatch_two / count
        test_acc_original = correct_original / count
        test_acc_discretized_4_bit = correct_discretized_4_bit / count
        test_acc_discretized_3_bit = correct_discretized_3_bit / count
        test_acc_rate = correct_rate / count
        print("Mismatch 1 test accuracy is %.4f Mismatch 2 test accuracy is %.4f Original test accuracy is %.4f Rate network test accuracy is %.4f Discretized 4bit test accuracy is %.4f Discretized 3bit test accuracy is %.4f" % (test_acc_mismatch_one, test_acc_mismatch_two, test_acc_original, test_acc_rate, test_acc_discretized_4_bit, test_acc_discretized_3_bit))


if __name__ == "__main__":

    # np.random.seed(42)

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--threshold', default=0.7, type=float, help="Threshold for prediction")
    parser.add_argument('--num_test', default=100, type=float, help="Number of test samples")
    parser.add_argument('--std', default=0.2, type=float, help="Percentage of mean for the mismatch standard deviation")
    parser.add_argument('--num-bits', default=4, type=int, help="Number of bits required to encode whole spectrum of weights")

    args = vars(parser.parse_args())
    verbose = args['verbose']
    threshold = args['threshold']
    num_test = args['num_test']
    mismatch_std = args['std']
    num_bits = args['num_bits']

    batch_size = 1
    percentage_data = 0.02
    # percentage_data = 1.0
    balance_ratio = 1.0
    snr = 10.

    for _ in range(10):


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
                                    mismatch_std=mismatch_std,
                                    num_bits=num_bits,
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
