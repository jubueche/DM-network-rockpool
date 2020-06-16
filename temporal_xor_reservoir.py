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
from rockpool.timeseries import TSContinuous
from rockpool import layers
from rockpool.layers import RecLIFCurrentInJax_IO
# from rockpool.layers.training import add_shim_lif_jax_sgd
from sklearn import metrics
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import argparse
from Utils import plot_matrices, filter_1d, generate_xor_sample
from jax.experimental.optimizers import sgd, adam
import jax.numpy as jnp
from jax import jit
from typing import Dict, Tuple, Any, Callable, Union, List, Optional
Params = Union[Dict, Tuple, List]
# from jax.config import config
# config.update("jax_debug_nans", True)

# - Change current directory to directory where this file is located
absolute_path = os.path.abspath(__file__)
directory_name = os.path.dirname(absolute_path)
os.chdir(directory_name)

class TemporalXORNetwork:
    def __init__(self,
                 num_neurons,
                 num_val,
                 num_test,
                 num_epochs,
                 samples_per_epoch,
                 verbose=0,
                 node_id=1,):

        self.verbose = verbose
        self.node_id = node_id
        self.dt = 0.001
        self.noise_std = 0.0
        self.duration = 1.0
        self.time_base = np.arange(0,self.duration,self.dt)

        self.num_val = num_val
        self.num_test = num_test

        self.num_epochs = num_epochs
        self.samples_per_epoch = samples_per_epoch
        
        # - Everything is stored in base_path/Resources/hey-snips/
        self.base_path = "/home/julian/Documents/dm-network-rockpool/"
        # self.base_path = "/home/julian_synsense_ai/dm-network-rockpool/"
        # - Every file saved by a node gets the prefix containing the node id
        self.node_prefix = "node_"+str(self.node_id)+str(int(np.abs(np.random.randn()*1e10)))
        
        # - Create NetworkADS
        model_path_net = os.path.join(self.base_path,"Resources/temporal-xor/x")

        if(os.path.exists(model_path_net)):
            print("Not implemented yet"); assert(False)
        else:
            self.num_neurons = num_neurons
            self.Nin = 1
            self.Nout = 1

            self.bias = -0.5
            self.tau_mem = 0.05
            self.tau_syn = np.random.uniform(0.03, 0.1, size=(self.num_neurons,))

            self.w_in = np.random.uniform(-3.0, 3.0, size=(self.Nin,self.num_neurons))
            self.w_rec = 0*np.random.uniform(-0.075, 0.075, size=(self.num_neurons,self.num_neurons))
            self.w_out = np.random.uniform(-3.0, 3.0, size=(self.num_neurons,self.Nout)) * self.tau_mem
            # self.w_out = np.copy(self.w_in.T) * self.tau_mem

            # - Initialize layer
            self.lyrRes = RecLIFCurrentInJax_IO(w_in = self.w_in,
                                                w_recurrent = self.w_rec,
                                                w_out = self.w_out,
                                                tau_mem = self.tau_mem,
                                                tau_syn = self.tau_syn,
                                                bias = self.bias,
                                                noise_std = self.noise_std,
                                                dt = self.dt)

            self.best_val_acc = 0.0
        # - End else create network

        self.best_model = self.lyrRes
        self.amplitude = 1

        # - Create dictionary for tracking information
        self.track_dict = {}
        self.track_dict["training_acc"] = []
        self.track_dict["validation_acc"] = []
        self.track_dict["testing_acc"] = 0.0

    def save(self, fn):
        return

    def get_data(self):
        data, target, _ = generate_xor_sample(total_duration=self.duration, dt=self.dt, amplitude=1.0)
        if((target > 0.5).any()):
            tgt_label = 1
        else:
            tgt_label = 0
        ts_data = TSContinuous(self.time_base, self.amplitude * data)
        ts_target = TSContinuous(self.time_base, target)
        return ts_data, ts_target, tgt_label

    def train(self):

        num_signal_iterations = 0

        if(self.verbose > 0):
            plt.figure(figsize=(8,5))
        
        time_horizon = 50
        avg_training_acc = np.zeros((time_horizon,))
        mses = np.ones((time_horizon,))
        time_track = []

        for epoch in range(self.num_epochs):

            epoch_loss = 0

            for batch_id in range(self.samples_per_epoch):

                (ts_data, ts_target, tgt_label) = self.get_data()

                # self.lyrRes.randomize_state()
                # self.lyrRes.reset_time()
                # plt.subplot(121)
                # ts_out = self.lyrRes.evolve(ts_data)
                # ts_out.plot()
                # ts_target.plot()
                # ts_data.plot()
                # plt.subplot(122)
                # self.lyrRes.spikes_last_evolution.plot()
                # plt.show()

                # plt.figure()
                # self.lyrRes.surrogate_last_evolution.plot()
                # plt.show()

                # - Do the training
                self.lyrRes.randomize_state()
                self.lyrRes.reset_time()
                is_first = (batch_id == 0) and (epoch == 0)

                _, _, _ = self.lyrRes.train_output_target(ts_data,
                                            ts_target,
                                            is_first = is_first,
                                            optimizer = sgd,
                                            opt_params={"step_size": 0.0001},
                                            loss_params={"lambda_mse": 1.0,
                                                         "reg_tau": 0.1,
                                                         "reg_l2_rec": 0.1,
                                                         "reg_l2_in": 0.1,
                                                         "reg_l2_out": 0.1,
                                                         "min_tau_syn": 0.01,
                                                         "min_tau_mem": 0.01},
                                            debug_nans = False)

                self.lyrRes.randomize_state()
                self.lyrRes.reset_time()
                ts_out = self.lyrRes.evolve(ts_data)
                final_out = filter_1d(ts_out.samples, alpha=0.95)
                ts_final_out = TSContinuous(self.time_base, final_out)

                if(verbose > 0):
                    plt.clf()
                    ts_target.plot(linestyle='--')
                    ts_data.plot()
                    ts_final_out.plot()
                    plt.draw()
                    plt.pause(0.00001)

                mse = np.mean((ts_target.samples - ts_final_out.samples)**2)
                mses[1:] = mses[:-1]
                mses[0] = mse
                if(is_first):
                    mses *= mse
                epoch_loss += mse
                
                if(ts_final_out.samples[np.argmax(np.abs(ts_out.samples))] > 0):
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
                mse_avg = np.mean(mses)
                time_track.append(num_signal_iterations)
                print("Target label", tgt_label, "Predicted label", predicted_label, ("Avg. training acc. %.4f" % (training_acc)), ("Avg. MSE %.4f" % (mse_avg)), ("Rec. L2: %.4f" % np.linalg.norm(self.lyrRes.weights.ravel())))
                print("--------------------",flush=True)

                # - Update tracking information
                self.track_dict["training_acc"].append(training_acc)
                
                num_signal_iterations += 1

            # Validate at the end of the epoch
            val_acc = self.perform_validation_set()

            # - Update tracking information
            self.track_dict["validation_acc"].append(val_acc)

            if(val_acc >= self.best_val_acc):
                self.best_val_acc = val_acc
                self.best_model = self.lyrRes
                # - Save in temporary file
                savedict = self.best_model.to_dict()
                savedict["best_val_acc"] = self.best_val_acc
                fn = os.path.join(self.base_path,("Resources/temporal-xor/"+self.node_prefix+"tmp_BPTT.json"))
                with open(fn, "w") as f:
                    json.dump(savedict, f)
                    print("Saved net",flush=True)

    def perform_validation_set(self):
        
        correct = 0
        counter = 0

        for batch_id in range(self.num_val):
            
            counter += 1

            # - Get input and target
            (ts_data, ts_target, tgt_label) = self.get_data()

            self.lyrRes.randomize_state()
            self.lyrRes.reset_time()
            ts_out = self.lyrRes.evolve(ts_data)
            final_out = filter_1d(ts_out.samples, alpha=0.95)
            ts_final_out = TSContinuous(self.time_base, final_out)
            
            if(self.verbose > 0):
                plt.clf()
                ts_final_out.plot(label="Spiking")
                ts_target.plot(label="Target")
                plt.legend()
                plt.draw()
                plt.pause(0.001)

            if(ts_final_out.samples[np.argmax(np.abs(ts_out.samples))] > 0):
                predicted_label = 1
            else:
                predicted_label = 0
            
            if(tgt_label == predicted_label):
                correct += 1

            print("--------------------------------",flush=True)
            print("VALIDATAION batch", batch_id,flush=True)
            print("true label", tgt_label, "pred label", predicted_label,flush=True)
            print("--------------------------------",flush=True)

        val_acc = correct / counter
        print("Validation accuracy is %.3f" % val_acc,flush=True)

        return val_acc


    def test(self):

        correct = 0
        counter = 0

        for batch_id in range(self.num_test):
            
            counter += 1

            # - Get input and target
            (ts_data, ts_target, tgt_label) = self.get_data()

            self.lyrRes.randomize_state()
            self.lyrRes.reset_time()
            ts_out = self.lyrRes.evolve(ts_data)
            final_out = filter_1d(ts_out.samples, alpha=0.95)
            ts_final_out = TSContinuous(self.time_base, final_out)
            
            if(self.verbose > 0):
                plt.clf()
                ts_final_out.plot(label="Spiking")
                ts_target.plot(label="Target")
                plt.legend()
                plt.draw()
                plt.pause(0.001)

            if(ts_final_out.samples[np.argmax(np.abs(ts_out.samples))] > 0):
                predicted_label = 1
            else:
                predicted_label = 0
            
            if(tgt_label == predicted_label):
                correct += 1

            print("--------------------------------",flush=True)
            print("TESTING batch", batch_id,flush=True)
            print("true label", tgt_label, "pred label", predicted_label,flush=True)
            print("--------------------------------",flush=True)

        test_acc = correct / counter
        print("Test accuracy is %.4f" % test_acc,flush=True)
        # - Save to tracking dict
        self.track_dict["testing_acc"] = test_acc

        # - Save the network
        param_string = "Resources/temporal-xor/"+self.node_prefix+"_test_acc"+str(test_acc)+"_BPTT.json"

        fn = os.path.join(self.base_path, param_string)
        # Save the model including the best validation score
        savedict = self.best_model.to_dict()
        savedict["best_val_acc"] = self.best_val_acc
        with open(fn, "w") as f:
            json.dump(savedict, f)


if __name__ == "__main__":

    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    
    parser.add_argument('--num', default=384, type=int, help="Number of neurons in the network")
    parser.add_argument('--verbose', default=0, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--epochs', default=200, type=int, help="Number of training epochs")
    parser.add_argument('--samples-per-epoch', default=1000, type=int, help="Number of training samples per epoch")
    parser.add_argument('--num-val', default=100, type=int, help="Number of validation samples")
    parser.add_argument('--num-test', default=1000, type=int, help="Number of test samples")
    parser.add_argument('--node-id', default=1, type=int, help="Node-ID")

    args = vars(parser.parse_args())
    num = args['num']
    verbose = args['verbose']
    num_epochs = args['epochs']
    samples_per_epoch = args['samples_per_epoch']
    num_val = args['num_val']
    num_test = args['num_test']
    node_id = args['node_id']

    model = TemporalXORNetwork(num_neurons=num,
                                num_val=num_val,
                                num_test=num_test,
                                num_epochs=num_epochs,
                                samples_per_epoch=samples_per_epoch,
                                verbose=verbose,
                                node_id=node_id)

    model.train()
    model.test()

    param_string = "Resources/temporal-xor/"+model.node_prefix+"_training_evolution_BPTT.json"
    fn = os.path.join(model.base_path,param_string)
    with open(fn, "w") as f:
        json.dump(model.track_dict, f)