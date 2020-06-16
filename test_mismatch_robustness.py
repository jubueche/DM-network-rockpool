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
        model_path_ads_net_1024 = "Resources/CloudModels/node_1_test_acc0.8401598401598401threshold0.7eta0.0001val_acc0.86tau_slow0.07tau_out0.07num_neurons1024.json"
        model_path_ads_net_768 = "Resources/hey-snips/node_7_test_acc0.8841158841158842threshold0.7eta0.0001no_fast_weights_val_acc0.9233870967741935tau_slow0.07tau_out0.07num_neurons768num_dist_weights-1.json"

        if(os.path.exists(model_path_ads_net_1024)):
            self.net_mismatch_1024 = NetworkADS.load(model_path_ads_net_1024)
            self.net_mismatch_768 = NetworkADS.load(model_path_ads_net_768)
            self.net_original_1024 = NetworkADS.load(model_path_ads_net_1024)
            self.net_original_768 = NetworkADS.load(model_path_ads_net_768)

            N1 = self.net_mismatch_1024.lyrRes.size
            N2 = self.net_mismatch_768.lyrRes.size
            Nc = self.net_mismatch_1024.lyrRes.out_size

            # - Test robustness: Use distribution for tau_fast, tau_out, tau_slow and V_thresh
            mean_tau_slow = self.net_mismatch_1024.lyrRes.tau_syn_r_slow
            mean_tau_fast = self.net_mismatch_1024.lyrRes.tau_syn_r_fast
            mean_tau_out = self.net_mismatch_1024.lyrRes.tau_syn_r_out
            mean_tau_mem = self.net_mismatch_1024.lyrRes.tau_mem

            # Apply first mismatch
            self.net_mismatch_1024.lyrRes.tau_syn_r_slow = np.abs(np.random.randn(N1)*self.mismatch_std*mean_tau_slow + mean_tau_slow)
            self.net_mismatch_1024.lyrRes.tau_syn_r_fast = np.abs(np.random.randn(N1)*self.mismatch_std*mean_tau_fast + mean_tau_fast)
            # self.net_mismatch_1024.lyrRes.tau_syn_r_out = np.abs(np.random.randn(Nc)*self.mismatch_std*mean_tau_out + mean_tau_out)
            self.net_mismatch_1024.lyrRes.tau_mem = np.abs(np.random.randn(N1)*self.mismatch_std*mean_tau_mem + mean_tau_mem)
            self.net_mismatch_1024.lyrRes.v_thresh = np.abs(np.random.randn(N1)*self.mismatch_std*np.mean(self.net_mismatch_1024.lyrRes.v_thresh) + np.mean(self.net_mismatch_1024.lyrRes.v_thresh))

            # Apply second mismatch
            self.net_mismatch_768.lyrRes.tau_syn_r_slow = np.abs(np.random.randn(N2)*self.mismatch_std*mean_tau_slow + mean_tau_slow)
            self.net_mismatch_768.lyrRes.tau_syn_r_fast = np.abs(np.random.randn(N2)*self.mismatch_std*mean_tau_fast + mean_tau_fast)
            # self.net_mismatch_768.lyrRes.tau_syn_r_out = np.abs(np.random.randn(Nc)*self.mismatch_std*mean_tau_out + mean_tau_out)
            self.net_mismatch_768.lyrRes.tau_mem = np.abs(np.random.randn(N2)*self.mismatch_std*mean_tau_mem + mean_tau_mem)
            self.net_mismatch_768.lyrRes.v_thresh = np.abs(np.random.randn(N2)*self.mismatch_std*np.mean(self.net_mismatch_768.lyrRes.v_thresh) + np.mean(self.net_mismatch_768.lyrRes.v_thresh))

            self.num_neurons = self.net_mismatch_1024.lyrRes.weights_fast.shape[0]
            self.tau_slow = self.net_mismatch_1024.lyrRes.tau_syn_r_slow
            self.tau_out = self.net_mismatch_1024.lyrRes.tau_syn_r_out
            self.amplitude = 50 / mean_tau_mem

            print("Loaded pretrained network from %s and %s" % (model_path_ads_net_1024, model_path_ads_net_768))
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

        correct_mismatch_1024 = 0
        correct_mismatch_768 = 0
        correct_original_1024 = 0
        correct_original_768 = 0
        correct_rate = 0
        count = 0

        for batch_id, [batch, test_logger] in enumerate(data_loader.test_set()):

            if batch_id > self.num_test:
                break

            audio_raw = batch[0][0]
            (ts_spiking_in, ts_rate_net_target_dynamics, ts_rate_out) = self.get_data(audio_raw=audio_raw)
            self.net_mismatch_1024.lyrRes.ts_target = ts_rate_net_target_dynamics
            self.net_mismatch_768.lyrRes.ts_target = ts_rate_net_target_dynamics
            self.net_original_1024.lyrRes.ts_target = ts_rate_net_target_dynamics
            self.net_original_768.lyrRes.ts_target = ts_rate_net_target_dynamics
            val_sim_mismatch_1024 = self.net_mismatch_1024.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1))
            val_sim_mismatch_768 = self.net_mismatch_768.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1))
            val_sim_original_1024 = self.net_original_1024.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1)); self.net_original_1024.reset_all()
            val_sim_original_768 = self.net_original_768.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1)); self.net_original_768.reset_all()

            out_val_mismatch_1024 = val_sim_mismatch_1024["output_layer"].samples.T
            out_val_mismatch_768 = val_sim_mismatch_768["output_layer"].samples.T
            out_val_original_1024 = val_sim_original_1024["output_layer"].samples.T
            out_val_original_768 = val_sim_original_768["output_layer"].samples.T

            self.net_mismatch_1024.reset_all()
            self.net_mismatch_768.reset_all()
            self.net_original_1024.reset_all()
            self.net_original_768.reset_all()
            
            # - Compute the final classification output for mismatch 1024
            final_out_mismatch_1024 = out_val_mismatch_1024.T @ self.w_out
            final_out_mismatch_1024 = filter_1d(final_out_mismatch_1024, alpha=0.95)

            # - Compute the final classification output for mismatch 768
            final_out_mismatch_768 = out_val_mismatch_768.T @ self.w_out
            final_out_mismatch_768 = filter_1d(final_out_mismatch_768, alpha=0.95)

            # - Compute the final classification output of original_1024 net
            final_out_original_1024 = out_val_original_1024.T @ self.w_out
            final_out_original_1024 = filter_1d(final_out_original_1024, alpha=0.95)

            # - Compute the final classification output of original_768 net
            final_out_original_768 = out_val_original_768.T @ self.w_out
            final_out_original_768 = filter_1d(final_out_original_768, alpha=0.95)

            # - Check for threshold crossing of mismatch 1024
            if ((final_out_mismatch_1024 > self.threshold).any()):
                predicted_label_mismatch_1024 = 1
            else:
                predicted_label_mismatch_1024 = 0
            # - Check for threshold crossing of mismatch 768
            if ((final_out_mismatch_768 > self.threshold).any()):
                predicted_label_mismatch_768 = 1
            else:
                predicted_label_mismatch_768 = 0
            # - Check for crossing of rate net
            if((ts_rate_out.samples > 0.7).any()):
                predicted_label_rate = 1
            else:
                predicted_label_rate = 0
            # - Check for crossing of the original_1024 net
            if ((final_out_original_1024 > self.threshold).any()):
                predicted_label_original_1024 = 1
            else:
                predicted_label_original_1024 = 0
            # - Check for crossing of the original_768 net
            if ((final_out_original_768 > self.threshold).any()):
                predicted_label_original_768 = 1
            else:
                predicted_label_original_768 = 0

            tgt_label = batch[0][1]
            if(predicted_label_mismatch_1024 == tgt_label):
                correct_mismatch_1024 += 1
            if(predicted_label_mismatch_768 == tgt_label):
                correct_mismatch_768 += 1
            if(predicted_label_rate == tgt_label):
                correct_rate += 1
            if(predicted_label_original_1024 == tgt_label):
                correct_original_1024 += 1
            if(predicted_label_original_768 == tgt_label):
                correct_original_768 += 1
            count += 1

            target = batch[0][2]
            target_times = np.arange(0, len(target) / self.fs, 1/self.fs)

            if(self.verbose > 0):
                plt.clf()
                plt.plot(np.arange(0,len(final_out_mismatch_1024)*self.dt, self.dt),final_out_mismatch_1024, label="Mismatch 1024")
                plt.plot(np.arange(0,len(final_out_mismatch_768)*self.dt, self.dt),final_out_mismatch_768, label="Mismatch 768")
                plt.plot(np.arange(0,len(final_out_original_1024)*self.dt, self.dt),final_out_original_1024, label="Original 1024")
                plt.plot(np.arange(0,len(final_out_original_768)*self.dt, self.dt),final_out_original_768, label="Original 768")
                plt.plot(target_times, target, label="Target")
                plt.plot(np.arange(0,len(ts_rate_out.samples)*self.dt, self.dt),ts_rate_out.samples, label="Rate")
                plt.axhline(y=self.threshold)
                plt.ylim([-0.5,1.0])
                plt.legend()
                plt.draw()
                plt.pause(0.001)

            print("--------------------------------")
            print("TESTING batch", batch_id)
            print("True label", tgt_label, "Mismatch-1024", predicted_label_mismatch_1024, "Mismatch-768", predicted_label_mismatch_768, "Original-1024", predicted_label_original_1024, "Original-768", predicted_label_original_768, "Rate label", predicted_label_rate)
            print("--------------------------------")

            test_logger.add_predictions(pred_labels=[predicted_label_mismatch_1024], pred_target_signals=[ts_rate_out.samples])
            fn_metrics('test', test_logger)

        test_acc_mismatch_1024 = correct_mismatch_1024 / count
        test_acc_mismatch_768 = correct_mismatch_768 / count
        test_acc_original_1024 = correct_original_1024 / count
        test_acc_original_768 = correct_original_768 / count
        test_acc_rate = correct_rate / count
        print("Mismatch 1024 test accuracy is %.4f Mismatch 768 test accuracy is %.4f Original_1024 test accuracy is %.4f Original_768 test accuracy is %.4f Rate network test accuracy is %.4f " % (test_acc_mismatch_1024, test_acc_mismatch_768, test_acc_original_1024, test_acc_original_768, test_acc_rate))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--num-test', default=100, type=float, help="Number of test samples")
    parser.add_argument('--std', default=0.2, type=float, help="Percentage of mean for the mismatch standard deviation")

    args = vars(parser.parse_args())
    verbose = args['verbose']
    num_test = args['num_test']
    mismatch_std = args['std']

    batch_size = 1
    percentage_data = 1.0
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
                                mismatch_std=mismatch_std,
                                num_epochs=0,
                                threshold=0.7,
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
