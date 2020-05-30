import warnings
warnings.filterwarnings('ignore')
from Utils import filter_1d
import time
import json
import numpy as np
import matplotlib
matplotlib.rc('font', family='Times New Roman')
matplotlib.rc('text',usetex=True)
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
        model_path_ads_net_fast = "Resources/hey-snips/node_8_test_acc0.8431568431568431threshold0.7eta0.0001val_acc0.8951612903225806tau_slow0.07tau_out0.07num_neurons768num_dist_weights-1_with_fast_conn.json"
        model_path_ads_net_no_fast = "Resources/hey-snips/node_7_test_acc0.8841158841158842threshold0.7eta0.0001no_fast_weights_val_acc0.9233870967741935tau_slow0.07tau_out0.07num_neurons768num_dist_weights-1.json"

        if(os.path.exists(model_path_ads_net_fast)):
            self.net_fast = NetworkADS.load(model_path_ads_net_fast)
            self.net_no_fast = NetworkADS.load(model_path_ads_net_no_fast)
            self.net_original_fast = NetworkADS.load(model_path_ads_net_fast)
            self.net_original_no_fast = NetworkADS.load(model_path_ads_net_no_fast)

            self.amplitude = 50 / np.mean(self.net_original_no_fast.lyrRes.tau_mem)

            print("Loaded pretrained network from %s and %s" % (model_path_ads_net_fast, model_path_ads_net_no_fast))
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

        correct_fast = 0
        correct_no_fast = 0
        correct_original_fast = 0
        correct_original_no_fast = 0
        correct_rate = 0
        count = 0
        t_start_suppress = 0.0
        t_stop_suppress = 5.0
        percentage_suppress = 0.3

        self.error_fast_list = []
        self.error_no_fast_list = []
        self.orig_error_fast_list = []
        self.orig_error_no_fast_list = []


        for batch_id, [batch, test_logger] in enumerate(data_loader.test_set()):

            if batch_id > self.num_test:
                break

            audio_raw = batch[0][0]
            (ts_spiking_in, ts_rate_net_target_dynamics, ts_rate_out) = self.get_data(audio_raw=audio_raw)
            self.net_fast.lyrRes.ts_target = ts_rate_net_target_dynamics
            self.net_no_fast.lyrRes.ts_target = ts_rate_net_target_dynamics
            self.net_original_fast.lyrRes.ts_target = ts_rate_net_target_dynamics
            self.net_original_no_fast.lyrRes.ts_target = ts_rate_net_target_dynamics

            # - Set the parameters for suppressing some neurons
            self.net_fast.lyrRes.t_start_suppress = t_start_suppress
            self.net_fast.lyrRes.t_stop_suppress = t_stop_suppress
            self.net_fast.lyrRes.percentage_suppress = percentage_suppress
            
            self.net_no_fast.lyrRes.t_start_suppress = t_start_suppress
            self.net_no_fast.lyrRes.t_stop_suppress = t_stop_suppress
            self.net_no_fast.lyrRes.percentage_suppress = percentage_suppress

            val_sim_fast = self.net_fast.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1))
            val_sim_no_fast = self.net_no_fast.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1))
            val_sim_original_fast = self.net_original_fast.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1)); self.net_original_fast.reset_all()
            val_sim_original_no_fast = self.net_original_no_fast.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1)); self.net_original_no_fast.reset_all()

            out_val_fast = val_sim_fast["output_layer"].samples.T
            out_val_no_fast = val_sim_no_fast["output_layer"].samples.T
            out_val_original_fast = val_sim_original_fast["output_layer"].samples.T
            out_val_original_no_fast = val_sim_original_no_fast["output_layer"].samples.T

            # - Compute the reconstruction accuracy
            target_val = ts_rate_net_target_dynamics.samples.T

            error_fast = np.sum(np.var(target_val-out_val_fast, axis=0, ddof=1)) / (np.sum(np.var(target_val, axis=0, ddof=1)))
            error_no_fast = np.sum(np.var(target_val-out_val_no_fast, axis=0, ddof=1)) / (np.sum(np.var(target_val, axis=0, ddof=1)))
            error_original_fast = np.sum(np.var(target_val-out_val_original_fast, axis=0, ddof=1)) / (np.sum(np.var(target_val, axis=0, ddof=1)))
            error_original_no_fast = np.sum(np.var(target_val-out_val_original_no_fast, axis=0, ddof=1)) / (np.sum(np.var(target_val, axis=0, ddof=1)))

            self.error_fast_list.append(error_fast)
            self.error_no_fast_list.append(error_no_fast)
            self.orig_error_fast_list.append(error_original_fast)
            self.orig_error_no_fast_list.append(error_original_no_fast)

            self.net_fast.reset_all()
            self.net_no_fast.reset_all()
            self.net_original_fast.reset_all()
            self.net_original_no_fast.reset_all()
            
            # - Compute the final classification output for  fast
            final_out_fast = out_val_fast.T @ self.w_out
            final_out_fast = filter_1d(final_out_fast, alpha=0.95)

            # - Compute the final classification output for  no_fast
            final_out_no_fast = out_val_no_fast.T @ self.w_out
            final_out_no_fast = filter_1d(final_out_no_fast, alpha=0.95)

            # - Compute the final classification output of original_fast net
            final_out_original_fast = out_val_original_fast.T @ self.w_out
            final_out_original_fast = filter_1d(final_out_original_fast, alpha=0.95)

            # - Compute the final classification output of original_no_fast net
            final_out_original_no_fast = out_val_original_no_fast.T @ self.w_out
            final_out_original_no_fast = filter_1d(final_out_original_no_fast, alpha=0.95)

            # - Check for threshold crossing of  fast
            if ((final_out_fast > self.threshold).any()):
                predicted_label_fast = 1
            else:
                predicted_label_fast = 0
            # - Check for threshold crossing of  no_fast
            if ((final_out_no_fast > self.threshold).any()):
                predicted_label_no_fast = 1
            else:
                predicted_label_no_fast = 0
            # - Check for crossing of rate net
            if((ts_rate_out.samples > 0.7).any()):
                predicted_label_rate = 1
            else:
                predicted_label_rate = 0
            # - Check for crossing of the original_fast net
            if ((final_out_original_fast > self.threshold).any()):
                predicted_label_original_fast = 1
            else:
                predicted_label_original_fast = 0
            # - Check for crossing of the original_no_fast net
            if ((final_out_original_no_fast > self.threshold).any()):
                predicted_label_original_no_fast = 1
            else:
                predicted_label_original_no_fast = 0

            tgt_label = batch[0][1]
            if(predicted_label_fast == tgt_label):
                correct_fast += 1
            if(predicted_label_no_fast == tgt_label):
                correct_no_fast += 1
            if(predicted_label_rate == tgt_label):
                correct_rate += 1
            if(predicted_label_original_fast == tgt_label):
                correct_original_fast += 1
            if(predicted_label_original_no_fast == tgt_label):
                correct_original_no_fast += 1
            count += 1

            target = batch[0][2]
            target_times = np.arange(0, len(target) / self.fs, 1/self.fs)

            if(self.verbose > 0):
                plt.clf()
                plt.plot(np.arange(0,len(final_out_fast)*self.dt, self.dt),final_out_fast, label=" fast")
                plt.plot(np.arange(0,len(final_out_no_fast)*self.dt, self.dt),final_out_no_fast, label=" no_fast")
                plt.plot(np.arange(0,len(final_out_original_fast)*self.dt, self.dt),final_out_original_fast, label="Original fast")
                plt.plot(np.arange(0,len(final_out_original_no_fast)*self.dt, self.dt),final_out_original_no_fast, label="Original no_fast")
                plt.plot(target_times, target, label="Target")
                plt.plot(np.arange(0,len(ts_rate_out.samples)*self.dt, self.dt),ts_rate_out.samples, label="Rate")
                plt.axhline(y=self.threshold)
                plt.ylim([-0.5,1.0])
                plt.legend()
                plt.draw()
                plt.pause(0.001)

            print("--------------------------------")
            print("TESTING batch", batch_id)
            print("Error fast", error_fast, "Error no fast", error_no_fast)
            print("Orig. Error fast", error_original_fast, "Orig. Error no fast", error_original_no_fast)
            print("True label", tgt_label, "-fast", predicted_label_fast, "-no_fast", predicted_label_no_fast, "Original-fast", predicted_label_original_fast, "Original-no_fast", predicted_label_original_no_fast, "Rate label", predicted_label_rate)
            print("--------------------------------")

            test_logger.add_predictions(pred_labels=[predicted_label_fast], pred_target_signals=[ts_rate_out.samples])
            fn_metrics('test', test_logger)

        test_acc_fast = correct_fast / count
        test_acc_no_fast = correct_no_fast / count
        test_acc_original_fast = correct_original_fast / count
        test_acc_original_no_fast = correct_original_no_fast / count
        test_acc_rate = correct_rate / count
        print(" fast test accuracy is %.4f  no_fast test accuracy is %.4f Original_fast test accuracy is %.4f Original_no_fast test accuracy is %.4f Rate network test accuracy is %.4f " % (test_acc_fast, test_acc_no_fast, test_acc_original_fast, test_acc_original_no_fast, test_acc_rate))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--num-test', default=100, type=float, help="Number of test samples")
    parser.add_argument('--std', default=0.2, type=float, help="Percentage of mean for the mismatch standard deviation")

    args = vars(parser.parse_args())
    verbose = args['verbose']
    num_test = args['num_test']

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

    with open("/home/julian/Documents/dm-network-rockpool/Resources/hey-snips/error_fast.npy", "wb") as f:
        np.save(f, model.error_fast_list)
    with open("/home/julian/Documents/dm-network-rockpool/Resources/hey-snips/error_no_fast.npy", "wb") as f:
        np.save(f, model.error_no_fast_list)
    with open("/home/julian/Documents/dm-network-rockpool/Resources/hey-snips/orig_error_fast.npy", "wb") as f:
        np.save(f, model.orig_error_fast_list)
    with open("/home/julian/Documents/dm-network-rockpool/Resources/hey-snips/orig_error_no_fast.npy", "wb") as f:
        np.save(f, model.orig_error_no_fast_list)

    # Plot over the course of training
    fig = plt.figure(figsize=(10,4))
    plt.plot(filter_1d(model.error_fast_list), color="C1", label="Error perturbed with fast conns")
    plt.plot(filter_1d(model.error_no_fast_list), color="C1", label="Error perturbed without fast conns",linestyle="--")
    plt.plot(filter_1d(model.orig_error_fast_list), color="C7", label="Error original with fast conns")
    plt.plot(filter_1d(model.orig_error_no_fast_list), color="C7", label="Error original without fast conns", linestyle="--")
    plt.legend(frameon=False, loc=3, prop={'size': 5})
    plt.show()