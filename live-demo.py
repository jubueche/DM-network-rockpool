import torch
import copy
import librosa
import sys
import time
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from aiCtxDemo import AiCtxMainWindow
from collections import deque
from torch.multiprocessing import Queue
from rockpool.timeseries import TSContinuous
from rockpool.networks import network
from scipy.special import softmax
from rockpool import layers
from rockpool.layers import ButterMelFilter
from rockpool.networks import NetworkADS
import json
from Utils import filter_1d

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


class MyDemo(AiCtxMainWindow):
    def __init__(self):

        # Demo parameters
        self.keywords = {
            "0": "Noise",
            "1": "Snips"
        }
        self.num_classes = len(self.keywords)
        self.max_time = 3.0
        self.fs = 16000
        self.fs_plot = 1600
        self.threshold = [1.0, 0.5]

        self.threshold_sum = 200

        # Filter parameters
        self.num_plot_filters = 16  # Num of frequency channels
        # minimal number of raw data from microphone needed
        # to generate a new filter output

        # # Model initialization
        # with open("demos/snips_best_scan.json", "r") as f:
        #     config_dict = json.load(f)
        #     layers_ = []
        #     for lyr_conf in config_dict['layers']:
        #         cls = getattr(layers, lyr_conf["class_name"])
        #         lyr_conf.pop("class_name")
        #         layers_.append(cls(**lyr_conf))
        #     self.lyr_filt, self.lyr_inp, self.lyr_res, self.lyr_out = layers_

        #     self.num_filters = self.lyr_filt.num_filters

        # - Load the rate model (for the input weight matrix)
        with open("Resources/hey-snips/rate_heysnips_tanh_0_16.model", "r") as f:
           config = json.load(f)
        self.w_in = np.array(config['w_in'])
        self.w_out = np.array(config['w_out'])
        self.num_filters = self.w_in.shape[0]
        # - Generate filter layer
        self.lyr_filt = ButterMelFilter(fs=self.fs,
                                num_filters=self.num_filters,
                                cutoff_fs=400.,
                                filter_width=2.,
                                num_workers=4,
                                name='filter')
        # - Load the ADS network
        model_path_ads_net = "Resources/hey-snips/node_7_test_acc0.8841158841158842threshold0.7eta0.0001no_fast_weights_val_acc0.9233870967741935tau_slow0.07tau_out0.07num_neurons768num_dist_weights-1.json"
        self.net = NetworkADS.load(model_path_ads_net)
        self.lyr_res = self.net.lyrRes
        self.out_layer = self.net.output_layer
        self.tau_mem = np.mean(self.net.lyrRes.tau_mem)
        self.amplitude = 50 / self.tau_mem
        self.net_dt = self.net.dt


        # Plotting parameters
        self.t = 0.0

        self.t_plot_net = 1. / 1000.
        self.t_plot_filt = 1. / 100.

        self.max_len_in = int(self.max_time * self.fs_plot)
        self.max_len_filt = int(self.max_time / self.t_plot_filt)
        self.max_len_net = int(self.max_time / self.t_plot_net)

        colors = {}
        cmap = plt.get_cmap("terrain")
        for i in range(self.num_classes):
            color = cmap(i / self.num_classes)
            colors[f"{i}"] = [int(v * 255) for v in color]
        self.model_colors = colors
        colors = {}
        cmap = plt.get_cmap("jet")
        for i in range(self.num_plot_filters):
            color = cmap(i / self.num_plot_filters)
            colors[f"{i}"] = [int(v * 255) for v in color]
        self.filter_colors = colors

        # Microphone data queue
        self.queue = Queue()
        self.data = deque(np.zeros(self.max_len_in), self.max_len_in)

        # Callback method
        def processdata(data, num_frames, t, status):
            self.queue.put(data)

        # Start microphone recording
        io_devices = sd.query_devices()
        for (i, device) in enumerate(io_devices):
            print(device["name"])
            # if "ALC1220 Analog" in device["name"]:
            if "default" in device["name"]:
                samplerate = device['default_samplerate']
                print(samplerate)
                uid = i
                #break
        self.downsample = int(samplerate/self.fs)
        self.microphone = sd.InputStream(
            device=uid, channels=1, samplerate=samplerate, callback=processdata)
        self.microphone.start()

        self.state = "SLEEPING"
        self.state_timer = 0.

        self.time_over_threshold = 0.
        self.filt_csum = np.zeros(self.num_filters)

        AiCtxMainWindow.__init__(self, strTitle="Network ADS")

    def initPlots(self):

        font=QtGui.QFont()
        font.setPixelSize(25)

        self.win = pg.GraphicsWindow()
        self.addWidgetToLayout(self.win)

        self.can_audio = self.win.addPlot(row=0, col=0)
        self.can_audio.setTitle("Raw Audio", size='30pt')
        self.can_audio.setRange(xRange=(0, self.max_time), yRange=(-1.0, 1.0))
        self.can_audio.enableAutoRange(x=False, y=False)
        self.can_audio.hideAxis("bottom")
        self.can_audio.getAxis("bottom").tickFont = font
        self.can_audio.getAxis("left").tickFont = font

        self.curve_input = self.can_audio.plot(pen={"width": 2})
        self.curve_input.setZValue(2)

        self.can_filter = self.win.addPlot(row=1, col=0)
        self.can_filter.setTitle("Butterworth Filter", size='30pt')
        self.can_filter.setRange(xRange=(0, self.max_time), yRange=(0, 0.4))
        self.can_filter.enableAutoRange(x=False, y=False)
        self.can_filter.hideAxis("bottom")
        self.can_audio.getViewBox().setXLink(self.can_filter.getViewBox())
        self.can_filter.getAxis("bottom").tickFont = font
        self.can_filter.getAxis("left").tickFont = font

        self.filter_curves = []
        for i in range(self.num_plot_filters):
            self.filter_curves.append(
                self.can_filter.plot(
                    pen={"width": 2, "color": self.filter_colors[f"{i}"]},
                )
            )

        self.command_info = pg.LabelItem("State", size="100pt")
        self.win.addItem(self.command_info, col=0, row=2)

        self.can_nw = self.win.addPlot(row=3, col=0)
        self.can_nw.setTitle("Network Output", size='30pt')
        self.can_nw.setRange(xRange=(0, self.max_time), yRange=(-0.2, 2 * self.threshold_sum))
        self.can_nw.enableAutoRange(x=False, y=False)
        self.can_nw.hideAxis("bottom")
        #self.can_nw.addLegend()
        self.can_filter.getViewBox().setXLink(self.can_nw.getViewBox())

        self.can_nw.getAxis("bottom").tickFont = font
        self.can_nw.getAxis("left").tickFont = font

        threshold_line = pg.InfiniteLine(
            pos=self.threshold_sum,
            angle=0,
            pen={"color": (255, 255, 255, 125), "style": QtCore.Qt.DashLine},
        )
        self.can_nw.getViewBox().addItem(threshold_line)

        self.model_curves = []
        for i in range(self.num_classes):
            self.model_curves.append(
                self.can_nw.plot(
                    pen={"width": 2, "color": self.model_colors[f"{i}"]},
                    name=f"{self.keywords[str(i)]}",
                )
            )

        self.can_energy = self.win.addPlot(row=4, col=0)
        self.can_energy.setTitle("Energy Consumption (uW)", size='30pt')
        self.can_energy.setRange(xRange=(0, self.max_time), yRange=(0., 300))
        self.can_energy.enableAutoRange(x=False, y=False)
        #self.can_energy.setLabel("bottom", text="Time", units="s")
        self.can_nw.getViewBox().setXLink(self.can_energy.getViewBox())
        self.can_energy.getAxis("bottom").tickFont = font
        self.can_energy.getAxis("left").tickFont = font

        self.curve_energy = self.can_energy.plot(pen={"width": 3})
        self.curve_energy.setZValue(2)

    def initData(self):
        self.t0 = 0
        self.time_data = np.linspace(self.t, self.t + self.max_time, self.max_len_in)
        self.input_data = deque(np.zeros(self.max_len_in), self.max_len_in)

        self.time_filt = np.linspace(self.t, self.t + self.max_time, self.max_len_filt)
        self.filter_out = [
            deque(np.zeros(self.max_len_filt), self.max_len_filt) for _ in range(self.num_plot_filters)]

        self.time_net = np.linspace(self.t, self.t + self.max_time, self.max_len_net)
        self.model_out = [
            deque(np.zeros(self.max_len_net), self.max_len_net) for _ in range(self.num_classes)]

        self.power_data = deque(np.zeros(self.max_len_net), self.max_len_net)

        self.peak_power = 0.
        self.avg_power = 0.
        self.power_hist = []


    def createHighlightItem(self, plot, color, coordinates):
        """
        Create a highlighting rectangle that can be drawn to a plot
            plot        plot element where highlight is drawn to
            color       highlight color
            coordinates highlight coordinates in plot's coord. system
        """
        color = copy.deepcopy(color)
        color[3] = 100

        r = pg.QtGui.QGraphicsRectItem(*coordinates)
        r.setPen(pg.mkPen(None))
        r.setBrush(pg.mkBrush(color))
        r.setZValue(-10)
        vb = plot.getViewBox()
        vb.addItem(r)
        return r, vb


    def calculate_power_consumption(self, inWeights, recWeights, tsEIn, tsERec, duration):
        en_spikegen = 200 * 1e-12 + 8 * 4 * 1e-12 + 14 * 1e-12
        en_core_broadcast = 120 * 1e-12
        en_core_route = 4 * 14 * 1e-12
        en_cam_match = 0

        #en_spikegen = 260 * 1e-12 + 507 * 1e-12
        #en_core_broadcast = 2.2 * 1e-9
        #en_core_route = 78 * 1e-12
        #en_cam_match = 26 * 1e-12

        fanout = np.sum(np.abs(recWeights), axis=1)
        spike_raster = tsERec.raster(dt=0.001, add_events=True)
        spike_counts = np.sum(spike_raster, axis=0)

        # - Cost for routing events depends on neuron population
        en_core_rec = 1 * en_core_broadcast + 0 * en_core_route
        en_cores = np.repeat(
            [en_core_rec], [recWeights.shape[0]]
        )

        en_per_spike = fanout * en_cam_match + en_cores + en_spikegen
        en_per_neuron = spike_counts * en_per_spike
        en_total = np.sum(en_per_neuron)

        power_dynamic = en_total / duration
        power_static = np.ceil(recWeights.shape[0] / 1024) * 210 * 1e-6

        power_static = 0.

        power_total = power_dynamic + power_static

        # - Include input events
        fanout_ext = np.sum(np.abs(inWeights), axis=1)
        en_per_ext_spike = (
                fanout_ext * en_cam_match + 4 * en_core_route +  0 * en_core_broadcast
        )

        # spike_raster_ext = tsEIn.raster(dt=0.001, add_events=True)
        # spike_counts_ext = np.sum(spike_raster_ext, axis=0)
        # en_per_ext = spike_counts_ext * en_per_ext_spike
        # en_ext_total = np.sum(en_per_ext)
        # power_ext = en_ext_total / duration
        # power_all = power_total + power_ext
        power_all = power_total

        return [power_all * 1e6]


    def smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def run_model(self, inp):
        try:
            times = (np.arange(len(inp)) / self.fs + self.lyr_filt.t)
            tsInput = TSContinuous(times, inp)
            times_net = np.arange(tsInput.t_start, tsInput.t_stop, self.net_dt)

            t0 = time.time()
            ts_filter = self.lyr_filt.evolve(tsInput)
            ts_inp = self.net.input_layer.evolve(TSContinuous(times_net, self.amplitude * ts_filter(times_net) @ self.w_in))
            
            self.lyr_res.ts_target = ts_inp
            ts_res = self.lyr_res.evolve(ts_inp)
            # ts_state = ts_res.append_c(ts_inp)
            ts_out = self.out_layer.evolve(ts_res)
            t1 = time.time()
            print(f"model eval {t1-t0}")

            net_dt = self.net.dt
            filt_dt = self.lyr_filt.dt

            #y_pred = dtSig['output'].samples
            #filt_out = dtSig['filter'].samples
            final_out = ts_out.samples @ self.w_out
            y_pred = filter_1d(final_out, alpha=0.95)

            filt_out = ts_filter.samples


            if((ts_res.channels > -1).any()):
                vfPower = copy.deepcopy(self.calculate_power_consumption(self.lyr_res.weights_in,
                                                                        self.lyr_res.weights_slow,
                                                                        ts_inp,
                                                                        ts_res,
                                                                        tsInput.duration))
            else:
                vfPower = [0.0]

            self.filt_csum *= 0.9
            self.filt_csum += np.cumsum(np.sum(filt_out, axis=0)) * 0.1

            for i in range(self.num_classes):
                tmp = y_pred
                tmp[np.where(tmp < self.threshold[1])[0]] = 0.
                if np.sum(tmp) > 0:
                    tmp[0] = self.model_out[i][-1]

                tmp = np.cumsum(tmp).tolist()
                self.model_out[i] += tmp

            self.power_data += vfPower * len(tmp)

            self.power_hist.append(vfPower[0] * tsInput.duration)
            self.peak_power = np.max([self.peak_power, vfPower[0]])

            for i in range(self.num_plot_filters):
                tmp = filt_out[::int(self.fs * self.t_plot_filt), int(i * self.num_filters / self.num_plot_filters)].tolist()
                tmp[0] = self.filter_out[i][-1]
                self.filter_out[i] += tmp


        except Exception as e:
            print(f"[inference] {e}")
        else:
            return y_pred, filt_out, vfPower

    def updateDisplay(self):
        """
        This method is to be called periodically to update the plots

        :return:
        """
        # Get data
        if self.queue.empty():
            return
        else:

            t0 = time.time()
            new_data = []
            while not self.queue.empty():
                new_data += np.ravel(self.queue.get()).tolist()

            # new_data = (np.array(new_data) * 2).tolist()

            new_data = librosa.core.resample(np.array(new_data), 44100, self.fs)
            plot_data = np.array(new_data)[::int(self.fs/self.fs_plot)]

            self.data += plot_data.tolist()

            t_step = len(new_data) / self.fs

            self.curve_input.setData(self.time_data, self.data)
            self.curve_energy.setData(self.time_net, self.power_data)

            # Pass through model
            t0 = time.time()
            self.run_model(new_data)
            t1 = time.time()

            # Plot filter output
            for i in range(self.num_plot_filters):
                self.filter_curves[i].setData(self.time_filt, self.filter_out[i])

            prob_class = []
            for i in range(self.num_classes):
                prob_ = np.array(list(self.model_out[i])[-int(0.4/self.t_plot_net):])
                prob_class.append(prob_)

            # Plot model output
            self.model_curves[1].setData(self.time_net, self.model_out[1])


            if self.state == "SLEEPING":
                if np.max(prob_class[1]) > self.threshold_sum:
                    self.state = "AWAKE"
                    self.state_timer = 1.0
            elif self.state == "AWAKE":
                if self.state_timer > 0:
                    self.state_timer -= t_step
                else:
                    self.state = "SLEEPING"


            print(self.state)
            self.command_info.setText(self.state)

            self.can_energy.getViewBox().setXRange(min(self.time_net), max(self.time_net))

            self.t += t_step
            self.time_data = np.linspace(self.t, self.t + self.max_time, self.max_len_in)
            self.time_filt = np.linspace(self.t, self.t + self.max_time, self.max_len_filt)
            self.time_net = np.linspace(self.t, self.t + self.max_time, self.max_len_net)

    def __del__(self):
        self.kill_all()

    def kill_all(self):
        print("Closing microphone recording")
        self.microphone.stop()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            print("Interrupting demo processes:")
            self.kill_all()
            sys.exit(0)


if __name__ == "__main__":

    app = QtGui.QApplication([])
    demo = MyDemo()
    timer = QtCore.QTimer()
    timer.timeout.connect(demo.updateDisplay)
    timer.start()
    demo.show()
    app.exec_()