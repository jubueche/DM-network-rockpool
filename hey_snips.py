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
from Utils import plot_matrices, filter_1d

import tracemalloc

# - Change current directory to directory where this file is located
absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)


class HeySnipsNetworkADS(BaseModel):
    def __init__(self,
                 labels,
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
                 dry_run=False,
                 discretize=-1,
                 discretize_dynapse=False,
                 node_id=1,
                 name="Snips ADS",
                 version="1.0"):
        
        super(HeySnipsNetworkADS, self).__init__(name,version)

        self.verbose = verbose
        self.dry_run = dry_run
        self.node_id = node_id
        self.fs = fs
        self.dt = 0.001

        self.num_val = num_val
        self.num_test = num_test

        self.num_epochs = num_epochs
        self.threshold = threshold

        self.N_filter = 1

        self.num_rate_neurons = 128 
        self.num_targets = len(labels)

        # - Everything is stored in base_path/Resources/hey-snips/
        self.base_path = "/home/julian/Documents/dm-network-rockpool/"
        # - Every file saved by a node gets the prefix containing the node id
        self.node_prefix = "node_"+str(self.node_id)

        rate_net_path = os.path.join(self.base_path, "Resources/hey-snips/rate_heysnips_tanh_0_16.model")
        with open(rate_net_path, "r") as f:
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
        self.num_channels = self.w_in.shape[0]

        self.lyr_filt = ButterMelFilter(fs=fs,
                                num_filters=self.num_channels,
                                cutoff_fs=400.,
                                filter_width=2.,
                                num_workers=4,
                                name='filter')


        # - Create NetworkADS
        model_path_ads_net = os.path.join(self.base_path,"Resources/hey-snips/x")

        if(os.path.exists(model_path_ads_net)):
            self.net = NetworkADS.load(model_path_ads_net)
            Nc = self.net.lyrRes.weights_in.shape[0]
            self.num_neurons = self.net.lyrRes.weights_fast.shape[0]
            self.tau_slow = self.net.lyrRes.tau_syn_r_slow
            self.tau_out = self.net.lyrRes.tau_syn_r_out
            self.tau_mem = np.mean(self.net.lyrRes.tau_mem)
            # Load best val accuracy
            with open(model_path_ads_net, "r") as f:
                loaddict = json.load(f)
                self.best_val_acc = loaddict["best_val_acc"]

            print("Loaded pretrained network from %s" % model_path_ads_net)
        else:
            Nc = self.num_rate_neurons
            self.num_neurons = num_neurons

            print("Building network with N: %d Nc: %d" % (self.num_neurons,Nc))

            lambda_d = 20
            lambda_v = 20
            self.tau_mem = 1/ lambda_v

            self.tau_slow = tau_slow
            self.tau_out = tau_out

            tau_syn_fast = 1e-3
            mu = 0.0005
            nu = 0.0001
            D = np.random.randn(Nc,self.num_neurons) / Nc
            # weights_in = D
            # weights_out = D.T
            weights_fast = (D.T@D + mu*lambda_d**2*np.eye(self.num_neurons))
            # - Start with zero weights 
            weights_slow = np.zeros((self.num_neurons,self.num_neurons))
  
            eta = eta
            k = 10 / self.tau_mem
            # noise_std = 0.0
            # - Pull out thresholds
            v_thresh = (nu * lambda_d + mu * lambda_d**2 + np.sum(abs(D.T), -1, keepdims = True)**2) / 2
            # v_reset = v_thresh - np.reshape(np.diag(weights_fast), (-1, 1))
            # v_rest = v_reset
            # - Fill the diagonal with zeros
            np.fill_diagonal(weights_fast, 0)

            # - Calculate weight matrices for realistic neuron settings
            v_thresh_target = 1.0*np.ones((self.num_neurons,)) # - V_thresh
            v_rest_target = 0.5*np.ones((self.num_neurons,)) # - V_rest = b

            b = v_rest_target
            a = v_thresh_target - b

            # - Feedforward weights: Divide each column i by the i-th threshold value and multiply by i-th value of a
            D_realistic = a*np.divide(D, v_thresh.ravel())
            weights_in_realistic = D_realistic
            weights_out_realistic = D_realistic.T
            weights_fast_realistic = a*np.divide(weights_fast.T, v_thresh.ravel()).T # - Divide each row

            weights_fast_realistic = np.zeros((self.num_neurons,self.num_neurons))

            # - Reset is given by v_reset_target = b - a
            v_reset_target = b - a
            noise_std_realistic = 0.00

            self.net = NetworkADS.SpecifyNetwork(N=self.num_neurons,
                                            Nc=Nc,
                                            Nb=self.num_neurons,
                                            weights_in=weights_in_realistic * self.tau_mem,
                                            weights_out= weights_out_realistic / 2,
                                            weights_fast= - weights_fast_realistic * self.tau_mem / tau_syn_fast,
                                            weights_slow = weights_slow,
                                            eta=eta,
                                            k=k,
                                            noise_std=noise_std_realistic,
                                            dt=self.dt,
                                            v_thresh=v_thresh_target,
                                            v_reset=v_reset_target,
                                            v_rest=v_rest_target,
                                            tau_mem=self.tau_mem,
                                            tau_syn_r_fast=tau_syn_fast,
                                            tau_syn_r_slow=self.tau_slow,
                                            tau_syn_r_out=self.tau_out,
                                            discretize=discretize,
                                            discretize_dynapse=discretize_dynapse,
                                            record=True
                                            )

            self.best_val_acc = 0.0
        # - End else create network

        self.best_model = self.net
        self.amplitude = 50 / self.tau_mem

        if(self.verbose > 0):
            plt.plot(self.best_model.lyrRes.v_thresh, label="V thresh")
            plt.plot(self.best_model.lyrRes.v_reset, label="V reset")
            plt.plot(self.best_model.lyrRes.v_rest,label="V rest")
            plt.legend()
            plt.show()

        # - Create dictionary for tracking information
        self.track_dict = {}
        self.track_dict["training_acc"] = []
        self.track_dict["training_recon_acc"] = []
        self.track_dict["validation_acc"] = []
        self.track_dict["validation_recon_acc"] = []
        self.track_dict["testing_acc"] = 0.0

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

    def train(self, data_loader, fn_metrics):

        self.best_model = self.net
        num_signal_iterations = 0

        if(self.verbose > 0):
            plt.figure(figsize=(8,5))

        # Create step schedule for k
        total_num_iter = data_loader.train_set.data.shape[0]*self.num_epochs
        # step_size = int(self.net.lyrRes.k_initial / 4)
        step_size = 25
        start_k = self.net.lyrRes.k_initial
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
            if(self.verbose > 0):
                plt.plot(np.arange(0,total_num_iter),f_k(np.arange(0,total_num_iter))); plt.title("Decay schedule for k"); plt.show()
        else:
            f_k = lambda t : 0

        # - Create schedule for eta
        a_eta = self.net.lyrRes.eta_initial
        b_eta = (total_num_iter/2) / np.log(100)
        c_eta = 0.0000001
        f_eta = lambda t,a_eta,b_eta : a_eta*np.exp(-t/b_eta) + c_eta

        if(self.verbose > 0):
            plt.plot(np.arange(0,total_num_iter),f_eta(np.arange(0,total_num_iter),a_eta,b_eta),label="Square root")
            plt.title("Decay schedule for eta"); plt.legend(); plt.show()

        time_horizon = 50
        recon_erros = np.ones((time_horizon,))
        avg_training_acc = np.zeros((time_horizon,)); avg_training_acc[:int(time_horizon/2)] = 1.0
        time_track = []

        for epoch in range(self.num_epochs):

            epoch_loss = 0

            for batch_id, [batch, train_logger] in enumerate(data_loader.train_set()):

                if(self.dry_run):
                    print("--------------------")
                    print("Epoch", epoch, "Batch ID", batch_id)
                    print("Target label", -1, "Predicted label", -1)
                    print("--------------------")
                    train_logger.add_predictions(pred_labels=[int(np.random.random())], pred_target_signals=[[0]])
                    fn_metrics('train', train_logger)
                    self.track_dict["training_acc"].append(np.random.random())
                    self.track_dict["training_recon_acc"].append(np.random.random())
                    continue
                # else ...

                # - Get the data from the batch
                audio_raw = batch[0][0]
                tgt_label = batch[0][1]
                (ts_spiking_in, ts_rate_net_target_dynamics, ts_rate_out) = self.get_data(audio_raw=audio_raw)

                if((ts_rate_out.samples > 0.7).any()):
                    predicted_label_rate = 1
                else:
                    predicted_label_rate = 0

                if(predicted_label_rate != tgt_label):
                    continue

                # - Do a training step
                self.net.lyrRes.is_training = True
                # - Adapt learning rate and k if it is the time
                # self.net.lyrRes.k = f_k(num_signal_iterations, a, b)
                self.net.lyrRes.k = f_k(num_signal_iterations)
                self.net.lyrRes.eta = f_eta(num_signal_iterations, a_eta, b_eta)

                # - Set the target
                self.net.lyrRes.ts_target = ts_rate_net_target_dynamics
                
                train_sim = self.net.evolve(ts_input=ts_spiking_in, verbose=(self.verbose==2))
                self.net.reset_all()
                self.net.lyrRes.is_training = False
                self.net.lyrRes.ts_target = None

                # - Compute train loss
                out_val = train_sim["output_layer"].samples.T
                target_val = ts_rate_net_target_dynamics.samples.T
                
                error = np.sum(np.var(target_val-out_val, axis=0, ddof=1)) / (np.sum(np.var(target_val, axis=0, ddof=1)))
                recon_erros[1:] = recon_erros[:-1]
                recon_erros[0] = error
                epoch_loss += error

                final_out = out_val.T @ self.w_out
                final_out = filter_1d(final_out, alpha=0.95)
                target = batch[0][2]
                target_times = np.arange(0, len(target) / self.fs, 1/self.fs)

                if(self.verbose > 0):
                    plt.clf()
                    plt.plot(np.arange(0,len(final_out)*self.dt, self.dt),final_out, label="Spiking")
                    plt.plot(target_times, target, label="Target")
                    plt.plot(np.arange(0,len(ts_rate_out.samples)*self.dt, self.dt),ts_rate_out.samples, label="Rate")
                    plt.axhline(y=self.threshold)
                    plt.ylim([-0.5,1.0])
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)

                if(np.any(final_out > self.threshold)):
                    predicted_label = 1
                else:
                    predicted_label = 0
                if(tgt_label == predicted_label):
                    correct = 1
                else:
                    correct = 0
                avg_training_acc[1:] = avg_training_acc[:-1]
                avg_training_acc[0] = correct

                print("--------------------",flush=True)
                print("Epoch", epoch, "Batch ID", batch_id,flush=True)
                training_acc = np.sum(avg_training_acc)/time_horizon
                reconstruction_acc = np.mean(recon_erros)
                time_track.append(num_signal_iterations)
                print("Target label", tgt_label, "Predicted label", predicted_label, ("Avg. training acc. %.4f" % (training_acc)), ("Avg. reconstruction error %.4f" % (reconstruction_acc)), "K", self.net.lyrRes.k,flush=True)
                print("--------------------",flush=True)

                train_logger.add_predictions(pred_labels=[predicted_label], pred_target_signals=[ts_rate_out.samples])
                fn_metrics('train', train_logger)

                # - Update tracking information
                self.track_dict["training_acc"].append(training_acc)
                self.track_dict["training_recon_acc"].append(reconstruction_acc)

                num_signal_iterations += 1

            yield {"train_loss": epoch_loss}

            # - Memory footprint
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage is {current / 10**6} MB; Peak was {peak / 10**6} MB; {peak / 10**9} GB",flush=True)

            # Validate at the end of the epoch
            val_acc, validation_recon_acc = self.perform_validation_set(data_loader=data_loader, fn_metrics=fn_metrics)

            # - Update tracking information
            self.track_dict["validation_acc"].append(val_acc)
            self.track_dict["validation_recon_acc"].append(validation_recon_acc)

            if(val_acc >= self.best_val_acc):
                self.best_val_acc = val_acc
                self.best_model = self.net
                # - Save in temporary file
                savedict = self.best_model.to_dict()
                savedict["best_val_acc"] = self.best_val_acc
                fn = os.path.join(self.base_path,("Resources/hey-snips/"+self.node_prefix+"tmp.json"))
                with open(fn, "w") as f:
                    json.dump(savedict, f)
                    print("Saved net",flush=True)
        
        if(total_num_iter > 0 and self.verbose > 0):
            _, ax1 = plt.subplots()
            l1 = ax1.plot(time_track, self.track_dict["training_recon_acc"], label="Reconstruction error")
            l2 = ax1.plot(time_track, self.track_dict["training_acc"], label="Training accuracy")
            ax2 = ax1.twinx()
            l3 = ax2.plot(np.arange(0,total_num_iter),f_k(np.arange(0,total_num_iter)), "k")
            lines = [l1[0],l2[0],l3[0]]
            ax2.legend(lines, ["Recon. error", "Training acc.", "K"])
            plt.show()

    def perform_validation_set(self, data_loader, fn_metrics):
        assert(self.net.lyrRes.is_training == False), "Validating, but is_training flag is set"
        assert(self.net.lyrRes.ts_target is None), "ts_target not set to None in spike_ads layer"

        errors = []
        correct = 0
        same_as_rate = 0
        counter = 0

        for batch_id, [batch, val_logger] in enumerate(data_loader.val_set()):
            if(batch_id >= self.num_val):
                break
            
            if(self.dry_run):
                print("--------------------------------")
                print("VALIDATAION batch", batch_id)
                print("true label", -1, "rate label", -1, "pred label", -1)
                print("--------------------------------")
                val_logger.add_predictions(pred_labels=[0], pred_target_signals=[[0]])
                fn_metrics('val', val_logger)
                counter += 1
                continue
            # else...

            counter += 1
            # - Get input and target
            audio_raw = batch[0][0]
            target = batch[0][2]
            target_times = np.arange(0, len(target) / self.fs, 1/self.fs)
            (ts_spiking_in, ts_rate_net_target_dynamics, ts_rate_out) = self.get_data(audio_raw=audio_raw)
            self.net.lyrRes.ts_target = ts_rate_net_target_dynamics
            val_sim = self.net.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 0 and batch_id < 3))
            out_val = val_sim["output_layer"].samples.T

            self.net.reset_all()

            target_val = ts_rate_net_target_dynamics.samples.T

            if(target_val.ndim == 1):
                target_val = np.reshape(target_val, (out_val.shape))
                target_val = target_val.T
                out_val = out_val.T

            err = np.sum(np.var(target_val-out_val, axis=0, ddof=1)) / (np.sum(np.var(target_val, axis=0, ddof=1)))
            errors.append(err)
            self.net.lyrRes.ts_target = None

            # - Compute the final classification output
            final_out = out_val.T @ self.w_out
            final_out = filter_1d(final_out, alpha=0.95)

            if(self.verbose > 0):
                plt.clf()
                plt.plot(np.arange(0,len(final_out)*self.dt, self.dt),final_out, label="Spiking")
                plt.plot(target_times, target, label="Target")
                plt.plot(np.arange(0,len(ts_rate_out.samples)*self.dt, self.dt),ts_rate_out.samples, label="Rate")
                plt.axhline(y=self.threshold)
                plt.ylim([-0.5,1.0])
                plt.legend()
                plt.draw()
                plt.pause(0.001)

            tgt_label = batch[0][1]
            if((final_out > self.threshold).any()):
                predicted_label = 1
            else:
                predicted_label = 0

            # - What did the rate network predict
            if((ts_rate_out.samples > 0.7).any()):
                rate_label = 1
            else:
                rate_label = 0
            
            if(tgt_label == predicted_label):
                correct += 1

            if(rate_label == predicted_label):
                same_as_rate += 1

            print("--------------------------------",flush=True)
            print("VALIDATAION batch", batch_id,flush=True)
            print("true label", tgt_label, "rate label", rate_label, "pred label", predicted_label,flush=True)
            print("--------------------------------",flush=True)

            val_logger.add_predictions(pred_labels=[predicted_label], pred_target_signals=[final_out])
            fn_metrics('val', val_logger)

        rate_acc = same_as_rate / counter
        val_acc = correct / counter
        print("Validation accuracy is %.3f | Compared to rate is %.3f" % (val_acc, rate_acc),flush=True)

        if(self.dry_run):
            return (np.random.random(),np.random.random())

        return (val_acc, np.mean(np.asarray(errors)))


    def test(self, data_loader, fn_metrics):

        correct = 0
        correct_rate = 0
        counter = 0
        for batch_id, [batch, test_logger] in enumerate(data_loader.test_set()):

            if batch_id > self.num_test:
                break

            if(self.dry_run):
                print("--------------------------------")
                print("TESTING batch", batch_id)
                print("true label", -1, "pred label", -1, "Rate label", -1)
                print("--------------------------------")
                test_logger.add_predictions(pred_labels=[0], pred_target_signals=[[0]])
                fn_metrics('test', test_logger)
                counter += 1
                continue

            audio_raw = batch[0][0]
            (ts_spiking_in, ts_rate_net_target_dynamics, ts_rate_out) = self.get_data(audio_raw=audio_raw)
            self.best_model.lyrRes.ts_target = ts_rate_net_target_dynamics
            val_sim = self.best_model.evolve(ts_input=ts_spiking_in, verbose=(self.verbose > 1))

            out_val = val_sim["output_layer"].samples.T
            self.best_model.reset_all()
            
            final_out = out_val.T @ self.w_out
            final_out = filter_1d(final_out, alpha=0.95)

            # check for threshold crossing
            if ((final_out > self.threshold).any()):
                predicted_label = 1
            else:
                predicted_label = 0
            if((ts_rate_out.samples > 0.7).any()):
                predicted_label_rate = 1
            else:
                predicted_label_rate = 0

            tgt_label = batch[0][1]
            if(predicted_label == tgt_label):
                correct += 1
            if(predicted_label_rate == tgt_label):
                correct_rate += 1
            counter += 1

            target = batch[0][2]
            target_times = np.arange(0, len(target) / self.fs, 1/self.fs)

            if(self.verbose > 0):
                plt.clf()
                plt.plot(np.arange(0,len(final_out)*self.dt, self.dt),final_out, label="Spiking")
                plt.plot(target_times, target, label="Target")
                plt.plot(np.arange(0,len(ts_rate_out.samples)*self.dt, self.dt),ts_rate_out.samples, label="Rate")
                plt.axhline(y=self.threshold)
                plt.ylim([-0.5,1.0])
                plt.legend()
                plt.draw()
                plt.pause(0.001)

            print("--------------------------------",flush=True)
            print("TESTING batch", batch_id,flush=True)
            print("true label", tgt_label, "pred label", predicted_label, "Rate label", predicted_label_rate,flush=True)
            print("--------------------------------",flush=True)

            test_logger.add_predictions(pred_labels=[predicted_label], pred_target_signals=[ts_rate_out.samples])
            fn_metrics('test', test_logger)

        test_acc = correct / counter
        test_acc_rate = correct_rate / counter
        print("Test accuracy is %.4f Rate network test accuracy is %.4f" % (test_acc, test_acc_rate),flush=True)
        # - Save to tracking dict
        self.track_dict["testing_acc"] = test_acc

        # - Save the network
        param_string = "Resources/hey-snips/"+self.node_prefix+"_test_acc"+str(test_acc)+"threshold"+str(self.threshold)+"eta" + \
                str(self.net.lyrRes.eta_initial)+"val_acc"+str(self.best_val_acc)+"tau_slow" + \
                str(self.tau_slow)+"tau_out"+str(self.tau_out)+"num_neurons"+str(self.num_neurons)+".json"

        fn = os.path.join(self.base_path, param_string)
        # Save the model including the best validation score
        savedict = self.best_model.to_dict()
        savedict["best_val_acc"] = self.best_val_acc
        with open(fn, "w") as f:
            json.dump(savedict, f)


if __name__ == "__main__":

    np.random.seed(42)

    # - Memory profiling
    tracemalloc.start()

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    
    parser.add_argument('--num', default=1024, type=int, help="Number of neurons in the network")
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--tau-slow', default=0.07, type=float, help="Time constant of slow recurrent synapses")
    parser.add_argument('--tau-out', default=0.07, type=float, help="Synaptic time constant of output synapses")
    parser.add_argument('--epochs', default=20, type=int, help="Number of training epochs")
    parser.add_argument('--threshold', default=0.7, type=float, help="Threshold for prediction")
    parser.add_argument('--eta', default=0.0001, type=float, help="Learning rate")
    parser.add_argument('--num-val', default=500, type=int, help="Number of validation samples")
    parser.add_argument('--num-test', default=1000, type=int, help="Number of test samples")
    parser.add_argument('--dry-run', default=False, action='store_true', help="Performs dry run of the simulation without doing any computation")
    parser.add_argument('--node-id', default=1, type=int, help="Node-ID")
    parser.add_argument('--percentage-data', default=0.02, type=float, help="Percentage of total training data used. Example: 0.02 is 2%.")
    parser.add_argument('--discretize', default=-1, type=int, help="Number of total different synaptic weights. -1 means no discretization. 8 means 3-bit precision.")
    parser.add_argument('--discretize-dynapse', default=False, action='store_true', help="Respect constraint of DYNAP-SE of having only 64 synapses per neuron. --discretize must not be -1.")

    args = vars(parser.parse_args())
    num = args['num']
    verbose = args['verbose']
    tau_slow = args['tau_slow']
    tau_out = args['tau_out']
    num_epochs = args['epochs']
    threshold = args['threshold']
    eta = args['eta']
    num_val = args['num_val']
    num_test = args['num_test']
    dry_run = args['dry_run']
    node_id = args['node_id']
    percentage_data = args['percentage_data']
    discretize_dynapse = args['discretize_dynapse']
    discretize = args['discretize']

    assert((not discretize_dynapse) or (discretize_dynapse and (discretize > 0))), "If --discretize-dynapse is specified, please choose a max. number of synapses per connection using the --discretize [int] field."

    batch_size = 1
    balance_ratio = 1.0
    snr = 10.

    experiment = HeySnipsDEMAND(batch_size=batch_size,
                                percentage=percentage_data,
                                snr=snr,
                                randomize_after_epoch=True,
                                is_tracking=False,
                                one_hot=False)

    num_train_batches = int(np.ceil(experiment.num_train_samples / batch_size))
    num_val_batches = int(np.ceil(experiment.num_val_samples / batch_size))
    num_test_batches = int(np.ceil(experiment.num_test_samples / batch_size))

    model = HeySnipsNetworkADS(labels=experiment._data_loader.used_labels,
                                num_neurons=num,
                                tau_slow=tau_slow,
                                tau_out=tau_out,
                                num_val=num_val,
                                num_test=num_test,
                                num_epochs=num_epochs,
                                threshold=threshold,
                                eta=eta,
                                verbose=verbose,
                                dry_run=dry_run,
                                discretize=discretize,
                                discretize_dynapse=discretize_dynapse,
                                node_id=node_id)

    experiment.set_model(model)
    experiment.set_config({'num_train_batches': num_train_batches,
                           'num_val_batches': num_val_batches,
                           'num_test_batches': num_test_batches,
                           'batch size': batch_size,
                           'percentage data': percentage_data,
                           'snr': snr,
                           'balance_ratio': balance_ratio})
    experiment.start()

    print("experiment done",flush=True)
    print(f"Accuracy score: {experiment.acc_scores}",flush=True)
    print("confusion matrix",flush=True)
    print(experiment.cm)
    param_string = "Resources/hey-snips/"+model.node_prefix+"_training_evolution.json"
    fn = os.path.join(model.base_path,param_string)
    with open(fn, "w") as f:
        json.dump(model.track_dict, f)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6} MB; Peak was {peak / 10**6} MB; {peak / 10**9} GB")
    tracemalloc.stop()

    # plt.subplot(311); plt.plot(np.arange(0,data_loader.train_set.data.shape[0]*len(self.track_dict['validation_recon_acc']),data_loader.train_set.data.shape[0]),self.track_dict['validation_recon_acc'],label="Recon error");
    # plt.plot(np.arange(0,data_loader.train_set.data.shape[0]*len(self.track_dict['validation_recon_acc']),data_loader.train_set.data.shape[0]),1-np.asarray(self.track_dict['validation_acc']),label="Val. Error")
    # plt.legend(); plt.xlabel("Epoch"); plt.xlim([0.0,total_num_iter]); plt.subplot(312);
    # plt.plot(1-np.asarray(self.track_dict['training_acc']),label="Training error"); plt.plot(self.track_dict['training_recon_acc'],label="Training recon error"); plt.legend(); plt.xlim([0,total_num_iter]);
    # plt.subplot(313); plt.plot(np.arange(0,total_num_iter),f_k(np.arange(0,total_num_iter)),label="k schedule"); plt.legend(); plt.xlabel("Num. signal iterations"); plt.show()