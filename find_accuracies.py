import numpy as np
import re
import os
import matplotlib
matplotlib.rc('font', family='Times New Roman')
matplotlib.rc('text',usetex=True)
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt

mismatch1_regex = r"(?:Mismatch 1 test accuracy is )([.\d+-]+)"
mismatch2_regex = r"(?:Mismatch 2 test accuracy is )([.\d+-]+)"
original_regex = r"(?:Original test accuracy is )([.\d+-]+)"
rate_net_regex = r"(?:Rate network test accuracy is )([.\d+-]+)"
disc_4bit_regex = r"(?:Discretized 4bit test accuracy is )([.\d+-]+)"
disc_3bit_regex = r"(?:Discretized 3bit test accuracy is )([.\d+-]+)"
disc_2bit_regex = r"(?:Discretized 2bit test accuracy is )([.\d+-]+)"

mismatch_1024_regex = r"(?:Mismatch 1024 test accuracy is )([.\d+-]+)"
mismatch_768_regex = r"(?:Mismatch 768 test accuracy is )([.\d+-]+)"
original_1024_regex = r"(?:Original_1024 test accuracy is )([.\d+-]+)"
original_768_regex = r"(?:Original_768 test accuracy is )([.\d+-]+)"

pattern_mismatch_1024 = re.compile(mismatch_1024_regex)
pattern_mismatch_768 = re.compile(mismatch_768_regex)
pattern_original_1024 = re.compile(original_1024_regex)
pattern_original_768 = re.compile(original_768_regex)

mismatch_1024_list = []
mismatch_768_list = []
original_1024_list = []
original_768_list = []

pattern_mismatch1 = re.compile(mismatch1_regex)
pattern_mismatch2 = re.compile(mismatch2_regex)
pattern_original = re.compile(original_regex)
pattern_rate = re.compile(rate_net_regex)
pattern_disc4 = re.compile(disc_4bit_regex)
pattern_disc3 = re.compile(disc_3bit_regex)
pattern_disc2 = re.compile(disc_2bit_regex)

mismatch_one_list = []
mismatch_two_list = []
original_list = []
rate_list = []
disc4_list = []
disc3_list = []
disc2_list = []

base_path = "/home/julian/Documents/dm-network-rockpool/Resources/hey-snips/"
files = os.listdir(base_path)
for fname in files:
    if(fname[:15] == "test_robustness"):
        file_path = os.path.join(base_path, fname)
        with open(file_path, 'r') as stream:
            for cnt,line in enumerate(stream):
                is_mismatch1 = pattern_mismatch1.search(line)
                is_mismatch2 = pattern_mismatch2.search(line)
                is_original = pattern_original.search(line)
                is_rate = pattern_rate.search(line)
                is_disc4 = pattern_disc4.search(line)
                is_disc3 = pattern_disc3.search(line)
                is_disc2 = pattern_disc2.search(line)

                if(is_mismatch1):
                    mismatch1_acc = float(is_mismatch1.group(1))
                    mismatch_one_list.append(mismatch1_acc)               
                if(is_mismatch2):
                    mismatch2_acc = float(is_mismatch2.group(1))
                    mismatch_two_list.append(mismatch2_acc)
                if(is_original):
                    original_acc = float(is_original.group(1))
                    original_list.append(original_acc)
                if(is_rate):
                    rate_acc = float(is_rate.group(1))
                    rate_list.append(rate_acc)
                if(is_disc4):
                    disc4_acc = float(is_disc4.group(1))
                    disc4_list.append(disc4_acc)
                if(is_disc3):
                    disc3_acc = float(is_disc3.group(1))
                    disc3_list.append(disc3_acc)
                if(is_disc2):
                    disc2_acc = float(is_disc2.group(1))
                    disc2_list.append(disc2_acc)

    elif(fname[:24] == "test_mismatch_robustness"):
        file_path = os.path.join(base_path, fname)
        with open(file_path, 'r') as stream:
            for cnt,line in enumerate(stream):
                is_mismatch_1024 = pattern_mismatch_1024.search(line)
                is_mismatch_768 = pattern_mismatch_768.search(line)
                is_original_1024 = pattern_original_1024.search(line)
                is_original_768 = pattern_original_768.search(line)

                if(is_mismatch_1024):
                    mismatch_1024_acc = float(is_mismatch_1024.group(1))
                    mismatch_1024_list.append(mismatch_1024_acc)
                if(is_mismatch_768):
                    mismatch_768_acc = float(is_mismatch_768.group(1))
                    mismatch_768_list.append(mismatch_768_acc)
                if(is_original_1024):
                    original_1024_acc = float(is_original_1024.group(1))
                    original_1024_list.append(original_1024_acc)
                if(is_original_768):
                    original_768_acc = float(is_original_768.group(1))
                    original_768_list.append(original_768_acc)

print(mismatch_1024_list)
print(mismatch_768_list)
print(original_1024_list)
print(original_768_list)

data_1 = [original_1024_list, mismatch_1024_list, original_768_list, mismatch_768_list]
# Multiple box plots on one Axes
fig = plt.figure(figsize=(6,2.3))
ax = plt.gca()
bplot = ax.boxplot(data_1, patch_artist=True,notch=False,boxprops=dict(facecolor="lightgreen", color="lightgreen"),flierprops=dict(markeredgecolor="r"))
ax.set_ylabel(r"Test accuracy")
ax.yaxis.grid(True)
num_boxes = 4
labels = ["Orig. 1024", "Mismatch 1024","Orig. 768","Mismatch 768"]
ax.set_xticklabels(labels,
                    rotation=0, fontsize=5)
medians = [np.median(l) for l in data_1]
pos = np.arange(num_boxes) + 1
upper_labels = [str(np.round(s, 2)) for s in medians]
weights = ['bold', 'semibold']
ax.set_ylim([0.5,1.0])
for tick, label in zip(range(num_boxes), ax.get_xticklabels()):
    k = tick % 2
    ax.text(pos[tick], .95, upper_labels[tick],
             transform=ax.get_xaxis_transform(),
             horizontalalignment='center', size='x-small',
             weight=weights[k])

plt.savefig("/home/julian/Documents/dm-network-rockpool/Latex/figures/figure8.png", dpi=1200)
plt.show()

print(mismatch_one_list)
print(mismatch_two_list)
print(original_list)
print(rate_list)
print(disc4_list)
print(disc3_list)
print(disc2_list)

print("Mean M1:",np.mean(mismatch_one_list), "STD:", np.std(mismatch_one_list))
print("Mean M2:",np.mean(mismatch_two_list), "STD:", np.std(mismatch_two_list))
print("Mean Original:",np.mean(original_list), "STD:", np.std(original_list))
print("Mean Rate:",np.mean(rate_list), "STD:", np.std(rate_list))
print("Mean disc4:",np.mean(disc4_list), "STD:", np.std(disc4_list))
print("Mean disc3:",np.mean(disc3_list), "STD:", np.std(disc3_list))
print("Mean disc2:",np.mean(disc2_list), "STD:", np.std(disc2_list))

data = [mismatch_one_list+mismatch_two_list, disc4_list, disc3_list, disc2_list, original_list, rate_list]

# Multiple box plots on one Axes
fig = plt.figure(figsize=(6,2.3))
ax = plt.gca()
ax.boxplot(data, patch_artist=True,notch=False,boxprops=dict(facecolor="lightgreen", color="lightgreen"),flierprops=dict(markeredgecolor="r"))
ax.yaxis.grid(True)

num_boxes = 6
lists = [mismatch_one_list+mismatch_two_list,disc4_acc,disc3_list,disc2_list,original_list,rate_list]
labels = ["Mismatch", "Disc. 4 bits","Disc. 3 bits","Disc. 2 bits", "Original", "Rate"]
ax.set_xticklabels(labels,
                    rotation=0, fontsize=5)
ax.set_ylabel(r"Test accuracy")
medians = [np.median(l) for l in lists]
pos = np.arange(num_boxes) + 1
upper_labels = [str(np.round(s, 2)) for s in medians]
weights = ['bold', 'semibold']
ax.set_ylim([0.5,1.0])
for tick, label in zip(range(num_boxes), ax.get_xticklabels()):
    k = tick % 2
    ax.text(pos[tick], .95, upper_labels[tick],
             transform=ax.get_xaxis_transform(),
             horizontalalignment='center', size='x-small',
             weight=weights[k])

plt.savefig("/home/julian/Documents/dm-network-rockpool/Latex/figures/figure7.png", dpi=1200)
plt.show()