import warnings
warnings.filterwarnings('ignore')
from Utils import running_mean, pISI_variance
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
from rockpool.layers import ButterMelFilter, RecRateEulerJax_IO, H_tanh, PassThrough
from rockpool.networks import NetworkADS
from sklearn import metrics
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
from tqdm import tqdm
import argparse
from Utils import filter_1d

# - Change current directory to directory where this file is located
absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)


class HeySnipsNetworkADS(BaseModel):
    def __init__(self,
                 labels,
                 validation_step,
                 num_val,
                 num_test,
                 num_epochs,
                 threshold,
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
        self.w_rec = np.array(config['w_recurrent'])
        self.w_out = np.array(config['w_out'])
        self.bias = config['bias']
        self.tau_rate = config['tau']

        self.rate_layer = RecRateEulerJax_IO(w_in=self.w_in,
                                             w_recurrent=self.w_rec,
                                             w_out=self.w_out,
                                             tau=self.tau_rate,
                                             bias=self.bias,
                                             activation_func=H_tanh,
                                             dt=self.dt,
                                             noise_std=0.0,
                                             name="hidden")

        self.lyr_filt = ButterMelFilter(fs=fs,
                                num_filters=self.w_in.shape[0],
                                cutoff_fs=400.,
                                filter_width=2.,
                                num_workers=4,
                                name='filter')


        # - Create NetworkADS
        self.netword_prefix = "test_acc0.8471760797342193threshold0.7eta2.2077035344734782e-06val_acc0.86tau_slow0.07tau_out0.07num_neurons1024"
        model_path_ads_net = "Resources/hey-snips/" + self.netword_prefix + ".json"

        if(os.path.exists(model_path_ads_net)):
            # Need to assign self.xxx variables
            self.net = NetworkADS.load(model_path_ads_net)
            self.num_neurons = self.net.lyrRes.weights_fast.shape[0]
            self.tau_slow = self.net.lyrRes.tau_syn_r_slow
            self.tau_out = self.net.lyrRes.tau_syn_r_out
            self.tau_mem = np.mean(self.net.lyrRes.tau_mem)
            # Load best val accuracy
            with open(model_path_ads_net, "r") as f:
                loaddict = json.load(f)
                self.best_val_acc = loaddict["best_val_acc"]
            print("Loaded pretrained network from %s" % model_path_ads_net)
        
            self.amplitude = 50 / self.net.lyrRes.tau_mem

        # - Create read-out layer that will be trained in the training method
        # initial_read_out_weights = 2*np.random.uniform(size=self.w_out.shape)-0.5
        initial_read_out_weights = self.w_out
        self.read_out_layer = PassThrough(weights=initial_read_out_weights, dt=self.dt)

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
            assert(self.net.lyrRes.is_training == False), "Validating, but is_training flag is set"
            assert(self.net.lyrRes.ts_target is None), "ts_target not set to None in spike_ads layer"
            correct = 0
            counter = 0
            for batch_id, [batch, val_logger] in enumerate(data_loader.val_set()):
                if(batch_id >= self.num_val):
                    break
                else:
                    counter += 1
                    # - Get input and target
                    audio_raw = batch[0][0]
                    (ts_spiking_in, ts_rate_net_target_dynamics, _) = self.get_data(audio_raw=audio_raw)
                    self.net.lyrRes.ts_target = ts_rate_net_target_dynamics
                    val_sim = self.net.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 0 and batch_id < 3)) ; self.net.reset_all()
                    ts_input_to_read_out = TSContinuous(np.arange(0, val_sim["output_layer"].samples.shape[0]*self.dt, self.dt), val_sim["output_layer"].samples)
                    ts_read_out = self.read_out_layer.evolve(ts_input_to_read_out) ; self.read_out_layer.reset_all()
                    # - Compute the final classification output
                    filtered_read_out = filter_1d(ts_read_out.samples, alpha=0.95)

                    tgt_label = batch[0][1]
                    if((filtered_read_out > self.threshold).any()):
                        predicted_label = 1
                    else:
                        predicted_label = 0
                    
                    if(tgt_label == predicted_label):
                        correct += 1
                    print("--------------------------------")
                    print("VALIDATAION batch", batch_id)
                    print("true label", tgt_label, "pred label", predicted_label)
                    print("--------------------------------")
                    val_logger.add_predictions(pred_labels=[predicted_label], pred_target_signals=[ts_read_out.samples])
                    fn_metrics('val', val_logger)
            val_acc = correct / counter
            print("Validation accuracy is %.3f" % val_acc)
            return val_acc


    def train(self, data_loader, fn_metrics):
        num_signal_iterations = 0
        _ = plt.figure(figsize=(10,5))
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            # train loop
            for batch_id, [batch, train_logger] in enumerate(data_loader.train_set()):

                is_first = (epoch == 0) and (batch_id == 0)
                is_last = (epoch == self.num_epochs-1) and (batch_id == data_loader.train_set.data.shape[0]-1)

                # if(num_signal_iterations % self.validation_step == 0):
                #     self.net.lyrRes.is_training == False
                #     self.net.lyrRes.ts_target = None
                #     self.perform_validation_set(data_loader=data_loader, fn_metrics=fn_metrics)

                # - Get the data from the batch
                audio_raw = batch[0][0]
                target = batch[0][2]
                times_target = np.arange(0, len(target) / self.fs, 1/self.fs)
                ts_target = TSContinuous(times_target, target)

                (ts_spiking_in, ts_rate_net_target_dynamics, _) = self.get_data(audio_raw=audio_raw)
                self.net.lyrRes.ts_target = ts_rate_net_target_dynamics

                # - Set the target
                train_sim = self.net.evolve(ts_input=ts_spiking_in, verbose=(self.verbose==2))
                self.net.reset_all()
                
                ts_input_to_read_out = TSContinuous(np.arange(0, train_sim["output_layer"].samples.shape[0]*self.dt, self.dt), train_sim["output_layer"].samples)

                # - Do the train step
                w_before = self.read_out_layer.weights
                self.read_out_layer.train_rr(ts_target=ts_target, ts_input=ts_input_to_read_out, regularize=0.0, is_first=is_first, is_last=is_last, train_biases=True)
                ts_read_out = self.read_out_layer.evolve(ts_input_to_read_out)
                self.read_out_layer.reset_all()
                print("Update norm is %.3f" % (np.linalg.norm(w_before-self.read_out_layer.weights)))

                filtered_read_out = filter_1d(ts_read_out.samples, alpha=0.95)

                plt.clf()
                plt.subplot(211)
                plt.plot(np.arange(0,len(filtered_read_out)*self.dt,self.dt), filtered_read_out)
                plt.plot(times_target, target)
                plt.ylim([-0.5,1.0])
                plt.subplot(212)
                plt.plot(self.read_out_layer.weights)
                plt.draw()
                plt.pause(0.001)

                tgt_label = batch[0][1]
                if(np.any(filtered_read_out > self.threshold)):
                    predicted_label = 1
                else:
                    predicted_label = 0
                print("--------------------------------")
                print("TRAINING batch", batch_id)
                print("true label", tgt_label, "pred label", predicted_label)
                print("--------------------------------")

                train_logger.add_predictions(pred_labels=[predicted_label], pred_target_signals=[ts_read_out.samples])
                fn_metrics('train', train_logger)

                num_signal_iterations += 1

            yield {"train_loss": epoch_loss}

    def test(self, data_loader, fn_metrics):

        correct = 0
        correct_rate = 0
        counter = 0
        for batch_id, [batch, test_logger] in enumerate(data_loader.test_set()):

            if batch_id > self.num_test:
                break
            else:
                counter += 1
                # - Get input and target
                audio_raw = batch[0][0]
                (ts_spiking_in, ts_rate_net_target_dynamics, ts_rate_out) = self.get_data(audio_raw=audio_raw)
                self.net.lyrRes.ts_target = ts_rate_net_target_dynamics
                test_sim = self.net.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 0 and batch_id < 3)) ; self.net.reset_all()
                ts_input_to_read_out = TSContinuous(np.arange(0, test_sim["output_layer"].samples.shape[0]*self.dt, self.dt), test_sim["output_layer"].samples)
                ts_read_out = self.read_out_layer.evolve(ts_input_to_read_out) ; self.read_out_layer.reset_all()
                # - Compute the final classification output

                filtered_read_out = filter_1d(ts_read_out.samples, alpha=0.95)
                
                tgt_label = batch[0][1]
                if((filtered_read_out > self.threshold).any()):
                    predicted_label = 1
                else:
                    predicted_label = 0
                if((ts_rate_out.samples > 0.7).any()):
                    predicted_label_rate = 1
                else:
                    predicted_label_rate = 0
                
                if(tgt_label == predicted_label_rate):
                    correct_rate += 1
                if(tgt_label == predicted_label):
                    correct += 1

                print("--------------------------------")
                print("TESTING batch", batch_id)
                print("true label", tgt_label, "pred label", predicted_label)
                print("--------------------------------")
                test_logger.add_predictions(pred_labels=[predicted_label], pred_target_signals=[ts_read_out.samples])
                fn_metrics('test', test_logger)
            

        test_acc = correct / counter
        test_acc_rate = correct_rate / counter
        print("Test accuracy is %.4f Rate network test accuracy is %.4f" % (test_acc, test_acc_rate))

        # Save the read-out layer
        savedict = self.read_out_layer.to_dict()
        fn = "Resources/hey-snips/" + self.netword_prefix + "_read_out.json"
        with open(fn, "w") as f:
            json.dump(savedict, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--epochs', default=20, type=int, help="Number of training epochs")
    parser.add_argument('--threshold', default=0.7, type=float, help="Threshold for prediction")
    parser.add_argument('--num_val', default=100, type=int, help="Number of validation samples")
    parser.add_argument('--val_after', default=50, type=int, help="Validate after number of signal iterations")
    parser.add_argument('--num_test', default=300, type=int, help="Number of test samples")

    args = vars(parser.parse_args())
    verbose = args['verbose']
    num_epochs = args['epochs']
    threshold = args['threshold']
    num_val = args['num_val']
    val_after = args['val_after']
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
                                validation_step=val_after,
                                num_val=num_val,
                                num_test=num_test,
                                num_epochs=num_epochs,
                                threshold=threshold,
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
