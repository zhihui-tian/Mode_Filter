import h5py
from scipy.io import savemat
h5_file = h5py.File(r"C:\Users\zhihui.tian\Downloads\mf_sz(2400x2400)_ng(20000)_nsteps(2000)_cov(25)_numnei(64)_cut(0)_gaussiansampling_tovishal.h5", 'r')
data_dict = {}
key_to_process = "sim0/grain_areas"
dataset = h5_file[key_to_process]
data_dict["GrainAllArea"] = dataset[()]  # Assign the dataset
savemat('GrainAllArea.mat', data_dict)
# Close the HDF5 file
h5_file.close()
# Print the dictionary to verify
print(data_dict)

import h5py
from scipy.io import savemat
h5_file = h5py.File(r"C:\Users\zhihui.tian\Downloads\mf_sz(2400x2400)_ng(20000)_nsteps(2000)_cov(25)_numnei(64)_cut(0)_gaussiansampling_tovishal.h5", 'r')
data_dict = {}
key_to_process = "sim0/grain_sides"
dataset = h5_file[key_to_process]
data_dict["GrainAllSides"] = dataset[()]  # Assign the dataset
savemat('GrainAllSides.mat', data_dict)
# Close the HDF5 file
h5_file.close()



import h5py
from scipy.io import savemat
h5_file = h5py.File(r"C:\Users\zhihui.tian\Downloads\mf_sz(2400x2400)_ng(20000)_nsteps(2000)_cov(25)_numnei(64)_cut(0)_gaussiansampling_tovishal.h5", 'r')
# data_dict = {}
key_to_process = "sim0/grain_areas_avg"
dataset = h5_file[key_to_process]
data_dict["GrainAreaAvg"] = dataset[()] # Assign the dataset
savemat('GrainAreaAvg.mat', data_dict)
# Close the HDF5 file
h5_file.close()


import h5py
from scipy.io import savemat

# Path to the HDF5 file
file_path = r"T:\MF_new\data\mf_sz(2400x2400)_ng(20000)_nsteps(2000)_cov(25)_numnei(32)_cut(0)_exp(Gaussian)_(5759999).h5"

# Keys to process and corresponding output filenames
keys_to_process = {
    "sim0/grain_areas": "GrainAllArea.mat",
    "sim0/grain_sides": "GrainAllSides.mat",
    "sim0/grain_areas_avg": "GrainAreaAvg.mat"
}

# Open the HDF5 file and process all keys
with h5py.File(file_path, 'r') as h5_file:
    for key, output_file in keys_to_process.items():
        # Extract the dataset and save it
        data_dict = {key.split('/')[-1]: h5_file[key][()]}  # Use the last part of the key as variable name
        savemat(output_file, data_dict)
        print(f"Saved {output_file}")



import h5py
from scipy.io import savemat

# Path to the HDF5 file
file_path = r"T:\MF_new\data\mf_sz(2400x2400)_ng(20000)_nsteps(2000)_cov(25)_numnei(8)_cut(0)_exp(Gaussian)_(5759999).h5"

# Keys to process and corresponding variable names and output filenames
keys_to_process = {
    "sim0/grain_areas": ("GrainAllArea", "GrainAllArea.mat"),
    "sim0/grain_sides": ("GrainAllSides", "GrainAllSides.mat"),
    "sim0/grain_areas_avg": ("GrainAreaAvg", "GrainAreaAvg.mat")
}

# Open the HDF5 file and process all keys
with h5py.File(file_path, 'r') as h5_file:
    for key, (var_name, output_file) in keys_to_process.items():
        # Extract the dataset and save it with the specified variable name
        data_dict = {var_name: h5_file[key][()]}
        savemat(output_file, data_dict)
        print(f"Saved {output_file} with variable '{var_name}'")


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

# Load distinguishable colors
from matplotlib.colors import to_rgb
from matplotlib.cm import get_cmap

def distinguishable_colors(n):
    cmap = get_cmap("tab20")
    return [cmap(i / n) for i in range(n)]

# Load data from .mat files
data_GrainAreaAvg = loadmat(r"C:\Users\zhihui.tian\Desktop\MF_new\MF-main\GrainAreaAvg.mat")
data_GrainAllArea = loadmat(r"C:\Users\zhihui.tian\Desktop\MF_new\MF-main\GrainAllArea.mat")
data_GrainAllSides = loadmat(r"C:\Users\zhihui.tian\Desktop\MF_new\MF-main\GrainAllSides.mat")

GrainAreaAvg = np.array(data_GrainAreaAvg["GrainAreaAvg"]).T
GrainAllArea = np.array(data_GrainAllArea["GrainAllArea"]).T
GrainAllSides = np.array(data_GrainAllSides["GrainAllSides"]).T

# Configuration
timeStep = 1000

# Validate timeStep range
if timeStep <= 1 or timeStep >= GrainAllArea.shape[1]:
    raise ValueError(f"Invalid timeStep: Ensure it is between 2 and {GrainAllArea.shape[1] - 1}")

# Slope from linear fit
slope = np.polyfit(np.arange(1, 1001), GrainAreaAvg[:1000, 0], 1)[0]

# Filter and normalize area data
A1 = GrainAllArea[:, timeStep].astype(np.float64)
A = A1[A1 > 0]
A2 = GrainAllArea[:, timeStep - 1].astype(np.float64)
A3 = GrainAllArea[:, timeStep + 1].astype(np.float64)
A_1 = A1[A1 > 0]
A_2 = A2[A1 > 0]
A_3 = A3[A1 > 0]
dA_dt_norm = (A_3 - A_2) / (2 * (np.pi / 3) * slope)

# Filter sides data
F1 = GrainAllSides[:, timeStep].astype(np.float64)
F = F1[A1 > 0]

# Distinguishable colors
ColorSet = distinguishable_colors(100)

# Figure 9: dA/dt vs F
Favg = []
dA_dt_normavg = []
dA_dt_normStd = []

for Fi in range(1, 13):
    Findex = np.where(F == Fi)[0]
    if len(Findex) > 0:
        Favg.append(Fi)
        dA_dt_normavg.append(np.mean(dA_dt_norm[Findex]))
        dA_dt_normStd.append(np.std(dA_dt_norm[Findex]))

plt.figure(9, figsize=(8, 6))
plt.errorbar(Favg, dA_dt_normavg, yerr=dA_dt_normStd, fmt='-s', markersize=10, linewidth=2, label='MFcov(25)_numnei(64)_cut(0)_exp(0)')
plt.plot(Favg, -(6 - np.array(Favg)), 'k-', linewidth=2, label='Analytical Eq (3)')
plt.xlabel('Number of sides F', fontweight='bold', fontsize=18, color='k')
plt.ylabel('{dA}/{dt}/({π/3} μ σ)', fontweight='bold', fontsize=18, color='k')
plt.axhline(0, color='k', linestyle='--', linewidth=2)
plt.axvline(6, color='k', linestyle='--', linewidth=2)
plt.legend(fontsize=14, loc='lower right', frameon=False)
plt.grid(True)
plt.tight_layout()
plt.savefig('MFMode_cov50_uniform_vNMR.png', dpi=600, transparent=True)

# Figure 1: dA/dt vs R/<R>
Rnorm = np.sqrt(A_1) / np.pi / np.mean(np.sqrt(A_1) / np.pi)

plt.figure(1, figsize=(8, 6))
for Fi in range(int(min(F)), int(max(F)) + 1):
    indices = np.where(F == Fi)[0]
    if len(indices) > 0:
        plt.scatter(Rnorm[indices], dA_dt_norm[indices], label=f'F = {Fi}', color=ColorSet[int(Fi)], s=30)

plt.xlabel('Normalized radius R/<R>', fontweight='bold', fontsize=18, color='k')
plt.ylabel('{dA}/{dt}/({π/3} μ σ)', fontweight='bold', fontsize=18, color='k')
plt.axhline(0, color='k', linestyle='--', linewidth=2)
plt.legend(fontsize=10, loc='best', frameon=False)
plt.grid(True)
plt.tight_layout()
# plt.savefig('MFModeModel_cov50_uniform_dAdtvsNormradius.png', dpi=600, transparent=True)

plt.show()
