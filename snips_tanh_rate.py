import matplotlib
matplotlib.use('Agg')
import time
import json
import numpy as np
import copy
import pylab as plt
from SIMMBA.BaseModel import BaseModel
from SIMMBA.experiments.HeySnipsDEMAND import HeySnipsDEMAND
from SIMMBA import BatchResult
from SIMMBA.metrics import accuracy, confusion_mat, roc, roc_auc
from rockpool.networks import network
from rockpool.timeseries import TSContinuous, TSEvent
from rockpool import layers
from rockpool.layers import FFLIFJax_IO, ButterMelFilter, FFIAFNest, RecLIFJax_IO, RecRateEulerJax_IO, H_tanh 
from sklearn import metrics
import itertools
from scipy.special import softmax
from scipy.signal import lfilter
from rockpool import layers
import torch
from torch import nn
import librosa
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from jax import jit
import jax.numpy as jnp
from typing import Dict, Tuple, Any, Callable, Union, List, Optional

Params = Union[Dict, Tuple, List]



@jit
def my_loss(
    params: Params,
    output_batch_t: jnp.ndarray,
    target_batch_t: jnp.ndarray,
    min_tau: float,
    lambda_mse: float = 1.0,
    reg_tau: float = 10000.0,
    reg_max_tau: float = 1.0,
    reg_l2_rec: float = 1.0,
    reg_diag_weights: float = 1.0,
    reg_bias: float = 1.0,
    reg_l2_w_out : float = 10000.0
) -> float:
    """
    Loss function for target versus output

    :param Params params:               Set of packed parameters
    :param jnp.ndarray output_batch_t:   Output rasterised time series [TxO]
    :param jnp.ndarray target_batch_t:   Target rasterised time series [TxO]
    :param float min_tau:               Minimum time constant
    :param float lambda_mse:            Factor when combining loss, on mean-squared error term. Default: 1.0
    :param float reg_tau:               Factor when combining loss, on minimum time constant limit. Default: 1e5
    :param float reg_max_tau:           Factor when combining loss, on maximum time constant. Default: 1.0
    :param float reg_l2_rec:            Factor when combining loss, on L2-norm term of recurrent weights. Default: 1.
    :param float reg_diag_weights:      Factor when combining loss, on diagonal enties of the recurrent weights. Default: 1.0
    :param float reg_bias:              Factor when combining loss, on biases. Default: 1.0

    :return float: Current loss value
    """
    # - Measure output-target loss
    mse = lambda_mse * jnp.mean((output_batch_t - target_batch_t) ** 2)

    # - Get loss for tau parameter constraints
    tau_loss = reg_tau * jnp.mean(
        jnp.where(params["tau"] < min_tau, jnp.exp(-(params["tau"] - min_tau)), 0)
    )

    # punish high time constants
    max_tau_loss = reg_max_tau * jnp.max(jnp.clip(params["tau"] - min_tau, 0, jnp.inf) ** 2)

    # punish high diag weights
    w_diag = params['w_recurrent'] * jnp.eye(len(params['w_recurrent']))
    loss_diag = reg_diag_weights * jnp.mean(jnp.abs(w_diag))

    # - Measure recurrent L2 norm
    w_res_norm = reg_l2_rec * jnp.mean(params["w_recurrent"] ** 2)

    # - Enforce uniform read-out weights
    # loss_w_out = reg_l2_w_out * (jnp.sum(jnp.abs(params["w_out"] / jnp.linalg.norm(params["w_out"])) - 0.2  ))

    # punish large biases
    loss_bias = reg_bias * jnp.mean(params['bias'] ** 2)

    # - Loss: target/output squared error, time constant constraint, recurrent weights norm, activation penalty
    fLoss = mse + tau_loss + w_res_norm + max_tau_loss + loss_bias + loss_diag # + loss_w_out

    # - Return loss
    return fLoss



class RNN(BaseModel):
    def __init__(self,
                 config,
                 labels, 
                 fs=16000.,
                 plot=False,
                 train=True,
                 num_val=np.inf,
                 name="HeySnips-RNN",
                 version="0.1"):

        super(RNN, self).__init__(name, version)

        plt.ioff()

        self.config = config
        self.plot = plot

        self.mov_avg_acc = 0.
        self.num_samples = 0

        self.fs = fs
        self.dt = 0.0001

        self.num_channels = 16 
        self.num_neurons = 128
        self.num_targets = 1 #len(labels) 

        self.num_epochs = 1

        self.mov_avg_acc = 0.
        self.num_samples = 0

        self.use_train = train
        self.num_val = num_val

        self.threshold = 0.7

        ##### CREATE NETWORK ######

        self.lyr_filt = ButterMelFilter(fs=fs,
                                        num_filters=self.num_channels,
                                        cutoff_fs=400.,
                                        filter_width=2.,
                                        num_workers=4,
                                        name='filter')


        if os.path.exists("rate_heysnips_tanh_0.model"):
            with open("rate_heysnips_tanh_0.model", "r") as f:
                config = json.load(f)

            w_in = np.array(config['w_in'])
            w_rec = np.array(config['w_recurrent'])
            w_out = np.array(config['w_out'])
            bias = config['bias']
            tau = config['tau']
        else:
            w_in = 10.0 * (np.random.rand(self.num_channels, self.num_neurons) - .5)
            w_rec = 0.2 * (np.random.rand(self.num_neurons, self.num_neurons) - .5)
            w_rec -= np.eye(self.num_neurons) * w_rec
            # w_out = 1.0 * (np.random.rand(self.num_neurons, self.num_targets) - .4)
            w_out = 0.4*np.random.uniform(size=(self.num_neurons, self.num_targets))-0.2
            bias = 0.0 * (np.random.rand(self.num_neurons) - 0.5)
            tau = np.linspace(0.01, 0.1, self.num_neurons)

            sr = np.max(np.abs(np.linalg.eigvals(w_rec)))
            w_rec = w_rec / sr * 0.95 

        self.lyr_hidden = RecRateEulerJax_IO(activation_func=H_tanh,
                                             w_in=w_in,
                                             w_recurrent=w_rec,
                                             w_out=w_out,
                                             tau=tau,
                                             bias=bias,
                                             dt=self.dt,
                                             noise_std=0.0,
                                             name="hidden")

        self.best_loss = float('inf')
        self.max_sample_length = 0.

    def terminate(self):
        pass

    def save(self, fn):
        if self.use_train:
            self.lyr_hidden.save_layer(fn)

    def plot_activity(self, ts_ext, ts_filt, ts_inp, ts_res, ts_out, ts_tgt):
        fig = plt.figure(figsize=[16, 10])
        ax1 = fig.add_subplot(5, 1, 1)
        if not ts_ext is None:
            ts_ext.plot()
        ax2 = fig.add_subplot(5, 1, 2, sharex=ax1)
        if not ts_filt is None:
            ts_filt.plot()
        ax3 = fig.add_subplot(5, 1, 3, sharex=ax1)
        if not ts_inp is None:
            ts_inp.plot()
        ax4 = fig.add_subplot(5, 1, 4, sharex=ax1)
        if not ts_res is None:
            ts_res.plot()
        ax5 = fig.add_subplot(5, 1, 5, sharex=ax1)
        ax5.set_prop_cycle(None)
        if not ts_out is None:
            ts_out.plot(linestyle='--')
        ax5.set_prop_cycle(None)
        if not ts_tgt is None:
            ts_tgt.plot()

        #plt.show(block=True)
        fig.savefig("activity_snips_tanh.png", dpi=300)
        plt.close('all')

    def predict(self, batch, evolve_hidden=True):

        self.lyr_filt.reset_time()
        self.lyr_hidden.reset_time()

        samples = np.hstack([s[0] for s in batch])
        tgt_signals = np.array([s[2] for s in batch][0])

        times_filt = np.arange(0, len(samples) / self.fs, 1/self.fs)

        ts_batch = TSContinuous(times_filt[:len(samples)], samples[:len(times_filt)])
        ts_tgt_batch = TSContinuous(times_filt[:len(tgt_signals)], tgt_signals[:len(times_filt)])

        ts_filt = self.lyr_filt.evolve(ts_batch)

        if np.any(np.isnan(ts_filt.samples)):
            print(1/0)
        ts_filt.samples[np.isnan(ts_filt.samples)] = 0

        if np.any(np.isnan(ts_tgt_batch.samples)):
            print(1/0)
        ts_tgt_batch.samples[np.isnan(ts_tgt_batch.samples)] = 0

        if evolve_hidden:
            ts_out = self.lyr_hidden.evolve(ts_filt)
        else:
            ts_out = None

        return ts_batch, ts_filt, ts_out, ts_tgt_batch



    def train(self, data_loader, fn_metrics):

        for epoch in range(self.num_epochs):

            epoch_loss = 0

            self.mov_avg_acc = 0.
            self.num_samples = 0

            # train loop
            for batch_id, [batch, train_logger] in enumerate(data_loader.train_set()):

                if not self.use_train:
                    break

                t0 = time.time()

                ts_ext, ts_filt, ts_out, ts_tgt = self.predict(batch, evolve_hidden=False)

                l_fcn, g_fcn, o_fcn = self.lyr_hidden.train_output_target(ts_filt,
                                                                            ts_tgt,
                                                                            is_first = (batch_id == 0) and (epoch == 0),
                                                                            opt_params={"step_size": 1e-3},
                                                                            loss_fcn=my_loss,
                                                                            loss_params={"lambda_mse": 1.0,
                                                                                        "reg_tau": 1000000.0,
                                                                                        "reg_l2_rec": 1.0,
                                                                                        "min_tau": 0.015,
                                                                                        "reg_max_tau": 1.0,
                                                                                        "reg_diag_weights": 1.0,
                                                                                        "reg_bias": 1000.0,
                                                                                        "reg_l2_w_out": 1000.0 })


                t1 = time.time()
                print(t1-t0)

                loss = np.array(l_fcn()).tolist()
                
                ts_out = self.lyr_hidden.evolve(ts_filt)

                if np.any(ts_out.samples > self.threshold):
                    predicted_label = 1
                else:
                    predicted_label = 0

                tgt_label = batch[0][1]

                pred_tgt_signals = ts_out.samples
                pred_tgt_signals[np.isnan(pred_tgt_signals)] = 0


                sr = np.max(np.abs(np.linalg.eigvals(self.lyr_hidden.w_recurrent)))
                print(f"spectral radius {sr}")

                w_diag = self.lyr_hidden.w_recurrent * np.eye(len(self.lyr_hidden.w_recurrent))
                print(f"diag weights {np.mean(np.abs(w_diag))}")

                print(f"bias max {np.max(self.lyr_hidden.bias)} mean {np.mean(self.lyr_hidden.bias)}")
                print(f"tau max {np.max(self.lyr_hidden.tau)} mean {np.mean(self.lyr_hidden.tau)}")
                print(f"w_out_max {np.max(np.abs(self.lyr_hidden.w_out))} mean {np.mean(self.lyr_hidden.w_out)}")

                if np.any(np.isnan(self.lyr_hidden.w_in)):
                    print("NAN in input weights", 1/0)

                if np.any(np.isnan(self.lyr_hidden.w_recurrent)):
                    print("NAN in recurrent weights", 1/0)

                if np.any(np.isnan(self.lyr_hidden.w_out)):
                    print("NAN in output weights", 1/0)

                self.mov_avg_acc = self.mov_avg_acc * self.num_samples + int(predicted_label == tgt_label)
                self.num_samples += 1
                self.mov_avg_acc /= self.num_samples

                if self.plot and batch_id % 100 == 0:
                    self.plot_activity(ts_ext, None, ts_filt, self.lyr_hidden.res_acts_last_evolution, 
                                       ts_out, ts_tgt)

                true_tgts = ts_tgt(ts_ext.times)
                pred_tgts = ts_out(ts_ext.times)

                true_tgts[np.isnan(true_tgts)] = 0.
                pred_tgts[np.isnan(pred_tgts)] = 0.

                mse = np.mean((true_tgts - pred_tgts) ** 2)
                epoch_loss += mse

                print("--------------------------------")
                print("epoch", epoch, "batch", batch_id, "MSE", mse, "epoch", epoch_loss)
                print("true label", tgt_label, "pred label", predicted_label, "loss", loss, "mvg acc", self.mov_avg_acc)
                print("--------------------------------")


                train_logger.add_predictions(pred_labels=[predicted_label], pred_target_signals=[pred_tgt_signals])
                fn_metrics('train', train_logger)

            val_loss = 0
            self.mov_avg_acc = 0.
            self.num_samples = 0

            # validation
            for batch_id, [batch, val_logger] in enumerate(data_loader.val_set()):

                if batch_id > self.num_val:
                    break

                ts_ext, ts_filt, ts_out, ts_tgt = self.predict(batch)
                
                if np.any(ts_out.samples > self.threshold):
                    predicted_label = 1
                else:
                    predicted_label = 0

                tgt_label = batch[0][1]

                pred_tgt_signals = ts_out.samples
                pred_tgt_signals[np.isnan(pred_tgt_signals)] = 0

                self.mov_avg_acc = self.mov_avg_acc * self.num_samples + int(predicted_label == tgt_label)
                self.num_samples += 1
                self.mov_avg_acc /= self.num_samples

                true_tgts = ts_tgt(ts_ext.times)
                pred_tgts = ts_out(ts_ext.times)

                true_tgts[np.isnan(true_tgts)] = 0.
                pred_tgts[np.isnan(pred_tgts)] = 0.

                mse = np.mean((true_tgts - pred_tgts) ** 2)
                val_loss += mse

                print("MSE", mse, "epoch", val_loss, "mvg acc", self.mov_avg_acc)

                val_logger.add_predictions(pred_labels=[predicted_label], pred_target_signals=[pred_tgt_signals])
                fn_metrics('val', val_logger)


            if val_loss < self.best_loss:
                self.save("rate_heysnips_tanh_0.model")
                self.best_loss = val_loss

            yield {"train_loss": epoch_loss, "val_loss": val_loss}


    def test(self, data_loader, fn_metrics):

        self.mov_avg_acc = 0.
        self.num_samples = 0

        test_true_labels = []
        test_pred_labels = []

        for batch_id, [batch, test_logger] in enumerate(data_loader.test_set()):

            if batch_id > self.num_val:
                break

            ts_ext, ts_filt, ts_out, ts_tgt = self.predict(batch)

            if np.any(ts_out.samples > self.threshold):
                predicted_label = 1
            else:
                predicted_label = 0


            tgt_label = batch[0][1]

            pred_tgt_signals = ts_out.samples
            pred_tgt_signals[np.isnan(pred_tgt_signals)] = 0

            self.mov_avg_acc = self.mov_avg_acc * self.num_samples + int(predicted_label == tgt_label)
            self.num_samples += 1
            self.mov_avg_acc /= self.num_samples

            print("test batch_id", batch_id, "mvg acc", self.mov_avg_acc)

            test_logger.add_predictions(pred_labels=[predicted_label], pred_target_signals=[pred_tgt_signals])
            fn_metrics('test', test_logger)

        test_acc = metrics.accuracy_score(test_true_labels, test_pred_labels)
        cm = metrics.confusion_matrix(test_true_labels, test_pred_labels)

        print(cm)

        print(f"test acc {test_acc}")



if __name__ == "__main__":

    config = {}

    print(config)

    batch_size = 1
    percentage_data = 1.0
    balance_ratio = 1.0
    snr = 10.

    experiment = HeySnipsDEMAND(batch_size=batch_size,
                                percentage=percentage_data,
                                snr=snr,
                                one_hot=False,
                                is_tracking=False)

    num_train_batches = int(np.ceil(experiment.num_train_samples / batch_size))
    num_val_batches = int(np.ceil(experiment.num_val_samples / batch_size))
    num_test_batches = int(np.ceil(experiment.num_test_samples / batch_size))

    model = RNN(config=config,
                labels=experiment._data_loader.used_labels, 
                plot=True,
                train=True,
                num_val=np.inf)


    experiment.set_model(model)
    experiment.set_config({'num_train_batches': num_train_batches,
                           'num_val_batches': num_val_batches,
                           'num_test_batches': num_test_batches,
                           'batch size': batch_size,
                           'percentage data': percentage_data,
                           'snr': snr,
                           'balance_ratio': balance_ratio,
                           'model_config': config})

    experiment.start()

    print("experiment done")




