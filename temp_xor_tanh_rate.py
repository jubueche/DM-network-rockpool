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
from Utils import generate_xor_sample
from rockpool.layers import RecRateEulerJax_IO, H_tanh
from rockpool.timeseries import TSContinuous
from jax import jit
import jax.numpy as jnp
from typing import Dict, Tuple, Any, Callable, Union, List, Optional

Params = Union[Dict, Tuple, List]

# - Define loss function
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
    reg_bias: float = 1.0
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

    # punish large biases
    loss_bias = reg_bias * jnp.mean(params['bias'] ** 2)

    # - Loss: target/output squared error, time constant constraint, recurrent weights norm, activation penalty
    fLoss = mse + tau_loss + w_res_norm + max_tau_loss + loss_bias + loss_diag

    # - Return loss
    return fLoss
# - End loss function

# - Define global parameters
verbose = 1
activation_func = H_tanh
duration = 1.0 # s
dt = 1e-3
amplitude = 1.0
num_units = 64
noise_std = 0.1
num_epochs = 500
num_batches = 100
num_channels = 1
num_targets = 1
time_base = np.arange(0,duration,dt)

# - Create the rate network
w_in = 10.0 * (np.random.rand(num_channels, num_units) - .5)
w_rec = 0.2 * (np.random.rand(num_units, num_units) - .5)
w_rec -= np.eye(num_units) * w_rec

# w_out = 1.0 * (np.random.rand(num_neurons, num_targets) - .4)
w_out = 0.4*np.random.uniform(size=(num_units, num_targets))-0.2
bias = 0.0 * (np.random.rand(num_units) - 0.5)
tau = np.linspace(0.01, 0.1, num_units)

sr = np.max(np.abs(np.linalg.eigvals(w_rec)))
w_rec = w_rec / sr * 0.95

lyr_hidden = RecRateEulerJax_IO(activation_func=activation_func,
                                        w_in=w_in,
                                        w_recurrent=w_rec,
                                        w_out=w_out,
                                        tau=tau,
                                        bias=bias,
                                        dt=dt,
                                        noise_std=noise_std,
                                        name="hidden")

# - Start training
for epoch in range(num_epochs):
    num_samples = 0
    mvg_avg_mse = 0

    for batch_id in range(num_batches):

        # - Generate new training data
        data, target = generate_xor_sample(total_duration=duration, dt=dt, amplitude=amplitude)
        ts_data = TSContinuous(time_base, data)
        ts_target = TSContinuous(time_base, target)
        
        lyr_hidden.reset_time()
        l_fcn, g_fcn, o_fcn = lyr_hidden.train_output_target(ts_data,
                                                                ts_target,
                                                                is_first = (batch_id == 0) and (epoch == 0),
                                                                opt_params={"step_size": 1e-4},
                                                                loss_fcn=my_loss,
                                                                loss_params={"lambda_mse": 1000000.0,
                                                                            "reg_tau": 1000000.0,
                                                                            "reg_l2_rec": 1.0,
                                                                            "min_tau": 0.015,
                                                                            "reg_max_tau": 1.0,
                                                                            "reg_diag_weights": 1.0,
                                                                            "reg_bias": 1000.0})

        ts_out = lyr_hidden.evolve(ts_data)

        if(verbose > 0):
            plt.clf()
            ts_target.plot(linestyle='--')
            ts_out.plot()
            plt.draw()
            plt.pause(0.001)

        mse = np.linalg.norm(ts_target.samples-ts_out.samples)**2
        mvg_avg_mse = mvg_avg_mse * num_samples + mse
        num_samples += 1
        mvg_avg_mse /= num_samples

        print(f"Moving average is {mvg_avg_mse}")
            
        # sr = np.max(np.abs(np.linalg.eigvals(lyr_hidden.w_recurrent)))
        # print(f"spectral radius {sr}")

        # w_diag = lyr_hidden.w_recurrent * np.eye(len(lyr_hidden.w_recurrent))
        # print(f"diag weights {np.mean(np.abs(w_diag))}")

        # print(f"bias max {np.max(lyr_hidden.bias)} mean {np.mean(lyr_hidden.bias)}")
        # print(f"tau max {np.max(lyr_hidden.tau)} mean {np.mean(lyr_hidden.tau)}")
        # print(f"w_out_max {np.max(np.abs(lyr_hidden.w_out))} mean {np.mean(lyr_hidden.w_out)}")
