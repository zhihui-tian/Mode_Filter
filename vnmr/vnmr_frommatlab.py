# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import loadmat
# import matplotlib as mpl
# from matplotlib.colors import ListedColormap, BoundaryNorm
# # Set global font to Times New Roman
# mpl.rcParams['font.family'] = 'Times New Roman'
# mpl.rcParams['mathtext.fontset'] = 'custom'
# mpl.rcParams['mathtext.rm'] = 'Times New Roman'
# mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
# mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'
#
# ### MCP
# data_grain_area_avg = loadmat("T:/MF_new/GrainAreaAvg_MCP_primme_0.66.mat")["GrainAreaAvg"]
# data_grain_all_area = loadmat("T:/MF_new/GrainAllArea_MCP_primme_0.66.mat")["GrainAllArea"]
# data_grain_all_sides = loadmat("T:/MF_new/GrainAllSides_MCP_primme_0.66.mat")["GrainAllSides"]
# time_step = 62
#
# ### cov 64
# data_grain_area_avg = loadmat("T:\MF_new\GrainAreaAvgGaussian12.mat")["GrainAreaAvg"]
# data_grain_all_area = loadmat("T:\MF_new\GrainAllAreaGaussian12.mat")["GrainAllArea"]
# data_grain_all_sides = loadmat("T:\MF_new\GrainAllSidesGaussian12.mat")["GrainAllSides"]
# timeStep=28
#
# data_grain_area_avg = loadmat("T:\MF_new\GrainAreaAvgGaussian_Square12.mat")["GrainAreaAvg"]
# data_grain_all_area = loadmat("T:\MF_new\GrainAllAreaGaussian_Square12.mat")["GrainAllArea"]
# data_grain_all_sides = loadmat("T:\MF_new\GrainAllSidesGaussian_Square12.mat")["GrainAllSides"]
# timeStep=21
#
# data_grain_area_avg = loadmat("T:\MF_new\GrainAreaAvgGaussian_Uniform12.mat")["GrainAreaAvg"]
# data_grain_all_area = loadmat("T:\MF_new\GrainAllAreaGaussian_Uniform12.mat")["GrainAllArea"]
# data_grain_all_sides = loadmat("T:\MF_new\GrainAllSidesGaussian_Uniform12.mat")["GrainAllSides"]
# timeStep=15
#
#
# ### cov 6
# data_grain_area_avg = loadmat("T:\MF_new\GrainAreaAvgGaussian6_64.mat")["GrainAreaAvg"]
# data_grain_all_area = loadmat("T:\MF_new\GrainAllAreaGaussian6_64.mat")["GrainAllArea"]
# data_grain_all_sides = loadmat("T:\MF_new\GrainAllSidesGaussian6_64.mat")["GrainAllSides"]
# timeStep=62
#
# data_grain_area_avg = loadmat("T:\MF_new\GrainAreaAvgGaussian_Square6_64.mat")["GrainAreaAvg"]
# data_grain_all_area = loadmat("T:\MF_new\GrainAllAreaGaussian_Square6_64.mat")["GrainAllArea"]
# data_grain_all_sides = loadmat("T:\MF_new\GrainAllSidesGaussian_Square6_64.mat")["GrainAllSides"]
# timeStep=49
#
# data_grain_area_avg = loadmat("T:\MF_new\GrainAreaAvgGaussian_Uniform6_64.mat")["GrainAreaAvg"]
# data_grain_all_area = loadmat("T:\MF_new\GrainAllAreaGaussian_Uniform6_64.mat")["GrainAllArea"]
# data_grain_all_sides = loadmat("T:\MF_new\GrainAllSidesGaussian_Uniform6_64.mat")["GrainAllSides"]
# timeStep=35
#
#
# # Transpose matrices for easier handling
# grain_area_avg = data_grain_area_avg.T
# grain_all_area = data_grain_all_area.T
# grain_all_sides = data_grain_all_sides.T
#
# # Calculate slope from polyfit
# pp = np.polyfit(np.arange(1, time_step * 2 + 1), grain_area_avg[:time_step * 2, 0], 1)
# slope = pp[0]
#
# # Process area data
# a1 = grain_all_area[:, time_step].astype(float)
# a2 = grain_all_area[:, time_step - 2].astype(float)
# a3 = grain_all_area[:, time_step + 2].astype(float)
#
# a_1 = a1[a1 > 0]
# a_2 = a2[a1 > 0]
# a_3 = a3[a1 > 0]
#
# dA_dt_norm = (a_3 - a_2) / 4 / (np.pi / 3) / slope
#
# # Process sides data
# f1 = grain_all_sides[:, time_step].astype(float)
# f = f1[a1 > 0]
#
# # Calculate averages and std deviations for plotting
# favg = []
# dA_dt_normavg = []
# dA_dt_normstd = []
#
# for fi in range(1, 13):
#     findex = np.where(f == fi)[0]
#     if findex.size > 0:
#         favg.append(fi)
#         dA_dt_normavg.append(np.mean(dA_dt_norm[findex]))
#         dA_dt_normstd.append(np.std(dA_dt_norm[findex]))
#
# favg = np.array(favg)
# dA_dt_normavg = np.array(dA_dt_normavg)
# dA_dt_normstd = np.array(dA_dt_normstd)
#
# # Plot error bars
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.errorbar(favg, dA_dt_normavg, yerr=dA_dt_normstd, fmt='-o', markersize=6, linewidth=2, capsize=3)
# ax.plot(favg, -(6 - favg), 'k-', linewidth=2)
#
# ax.set_xlabel('Number of sides F', fontsize=18, labelpad=10)
# ax.set_ylabel(r'$\frac{dA}{dt}/\left(\frac{\pi}{3} \mu \sigma\right)$', fontsize=18, labelpad=10)
# ax.tick_params(axis='both', labelsize=14)
# ax.legend(fontsize=14)
# ax.axhline(0, color='k', linestyle='--', linewidth=1.5)
# ax.axvline(6, color='k', linestyle='--', linewidth=1.5)
#
# plt.tight_layout()
#
# # Second plot for normalized radius
# r_norm = np.sqrt(a_1) / np.pi / np.mean(np.sqrt(a_1) / np.pi)
#
# unique_f = np.unique(f)
# n_colors = len(unique_f)
#
# # Create a discrete colormap for unique sides
# colors = plt.cm.tab20(np.linspace(0, 1, n_colors))  # Generate discrete colors
# cmap = ListedColormap(colors)
# bounds = unique_f - 0.5
# bounds = np.append(bounds, unique_f[-1] + 0.5)  # Extend bounds to include max
# norm = BoundaryNorm(bounds, cmap.N)
#
# # Plot scatter with discrete colormap
# fig, ax = plt.subplots(figsize=(10, 6))
# scatter = ax.scatter(r_norm, dA_dt_norm, c=f, cmap=cmap, norm=norm, s=3)  # Smaller scatter size
#
# # Add colorbar with mapping to unique sides
# cbar = plt.colorbar(scatter, ax=ax, boundaries=bounds, ticks=unique_f, pad=0.02)
# cbar.set_label('Number of sides F', fontsize=14)
# cbar.ax.tick_params(labelsize=12)
#
# # Set plot labels
# ax.set_xlabel(r'Normalized radius $R/\langle R \rangle$', fontsize=18, labelpad=10)
# ax.set_ylabel(r'$\frac{dA}{dt}/\left(\frac{\pi}{3} \mu \sigma\right)$', fontsize=18, labelpad=10)
# ax.tick_params(axis='both', labelsize=14)
#
# plt.tight_layout()
# plt.show()




# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import loadmat
# import matplotlib as mpl
# from matplotlib.colors import ListedColormap, BoundaryNorm
#
# # Set global font to Times New Roman
# mpl.rcParams['font.family'] = 'Times New Roman'
# mpl.rcParams['mathtext.fontset'] = 'custom'
# mpl.rcParams['mathtext.rm'] = 'Times New Roman'
# mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
# mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'
#
# # Define datasets and their corresponding time steps
# datasets = [
#     {"name": "MCP_primme_0.66", "path": "T:/MF_new/GrainAreaAvg_MCP_primme_0.66.mat", "time_step": 62},
#     {"name": "Gaussian12", "path": "T:/MF_new/GrainAreaAvgGaussian12.mat", "time_step": 28},
#     {"name": "Gaussian_Square12", "path": "T:/MF_new/GrainAreaAvgGaussian_Square12.mat", "time_step": 21},
#     {"name": "Gaussian_Uniform12", "path": "T:/MF_new/GrainAreaAvgGaussian_Uniform12.mat", "time_step": 15},
#     {"name": "Gaussian6_64", "path": "T:/MF_new/GrainAreaAvgGaussian6_64.mat", "time_step": 62},
#     {"name": "Gaussian_Square6_64", "path": "T:/MF_new/GrainAreaAvgGaussian_Square6_64.mat", "time_step": 49},
#     {"name": "Gaussian_Uniform6_64", "path": "T:/MF_new/GrainAreaAvgGaussian_Uniform6_64.mat", "time_step": 35},
# ]
#
# # Loop through each dataset
# for dataset in datasets:
#     # Load data
#     data_grain_area_avg = loadmat(dataset["path"])["GrainAreaAvg"]
#     data_grain_all_area = loadmat(dataset["path"].replace("GrainAreaAvg", "GrainAllArea"))["GrainAllArea"]
#     data_grain_all_sides = loadmat(dataset["path"].replace("GrainAreaAvg", "GrainAllSides"))["GrainAllSides"]
#     time_step = dataset["time_step"]
#
#     # Transpose matrices for easier handling
#     grain_area_avg = data_grain_area_avg.T
#     grain_all_area = data_grain_all_area.T
#     grain_all_sides = data_grain_all_sides.T
#
#     # Calculate slope from polyfit
#     pp = np.polyfit(np.arange(1, time_step * 2 + 1), grain_area_avg[:time_step * 2, 0], 1)
#     slope = pp[0]
#
#     # Process area data
#     a1 = grain_all_area[:, time_step].astype(float)
#     a2 = grain_all_area[:, time_step - 2].astype(float)
#     a3 = grain_all_area[:, time_step + 2].astype(float)
#
#     a_1 = a1[a1 > 0]
#     a_2 = a2[a1 > 0]
#     a_3 = a3[a1 > 0]
#
#     dA_dt_norm = (a_3 - a_2) / 4 / (np.pi / 3) / slope
#
#     # Process sides data
#     f1 = grain_all_sides[:, time_step].astype(float)
#     f = f1[a1 > 0]
#
#     # Calculate averages and std deviations for plotting
#     favg = []
#     dA_dt_normavg = []
#     dA_dt_normstd = []
#
#     for fi in range(1, 13):
#         findex = np.where(f == fi)[0]
#         if findex.size > 0:
#             favg.append(fi)
#             dA_dt_normavg.append(np.mean(dA_dt_norm[findex]))
#             dA_dt_normstd.append(np.std(dA_dt_norm[findex]))
#
#     favg = np.array(favg)
#     dA_dt_normavg = np.array(dA_dt_normavg)
#     dA_dt_normstd = np.array(dA_dt_normstd)
#
#     # Plot error bars
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.errorbar(favg, dA_dt_normavg, yerr=dA_dt_normstd, fmt='-o', markersize=6, linewidth=2, capsize=3)
#     ax.plot(favg, -(6 - favg), 'k-', linewidth=2)
#
#     ax.set_xlabel('Number of sides F', fontsize=18, labelpad=10)
#     ax.set_ylabel(r'$\frac{dA}{dt}/\left(\frac{\pi}{3} \mu \sigma\right)$', fontsize=18, labelpad=10)
#     ax.tick_params(axis='both', labelsize=14)
#     ax.axhline(0, color='k', linestyle='--', linewidth=1.5)
#     ax.axvline(6, color='k', linestyle='--', linewidth=1.5)
#
#     plt.tight_layout()
#     plt.savefig(f"{dataset['name']}_errorbar.png")  # Save error bar figure
#     plt.close()
#
#     # Second plot for normalized radius
#     r_norm = np.sqrt(a_1) / np.pi / np.mean(np.sqrt(a_1) / np.pi)
#
#     unique_f = np.unique(f)
#     n_colors = len(unique_f)
#
#     # Create a discrete colormap for unique sides
#     colors = plt.cm.tab20(np.linspace(0, 1, n_colors))  # Generate discrete colors
#     cmap = ListedColormap(colors)
#     bounds = unique_f - 0.5
#     bounds = np.append(bounds, unique_f[-1] + 0.5)  # Extend bounds to include max
#     norm = BoundaryNorm(bounds, cmap.N)
#
#     # Plot scatter with discrete colormap
#     fig, ax = plt.subplots(figsize=(8, 6))
#     scatter = ax.scatter(r_norm, dA_dt_norm, c=f, cmap=cmap, norm=norm, s=3)  # Smaller scatter size
#
#     # Add colorbar with mapping to unique sides
#     cbar = plt.colorbar(scatter, ax=ax, boundaries=bounds, ticks=unique_f, pad=0.02)
#     cbar.set_label('Number of sides F', fontsize=14)
#     cbar.ax.tick_params(labelsize=12)
#
#     # Set plot labels
#     ax.set_xlabel(r'Normalized radius $R/\langle R \rangle$', fontsize=18, labelpad=10)
#     ax.set_ylabel(r'$\frac{dA}{dt}/\left(\frac{\pi}{3} \mu \sigma\right)$', fontsize=18, labelpad=10)
#     ax.tick_params(axis='both', labelsize=14)
#
#     plt.tight_layout()
#     plt.savefig(f"{dataset['name']}_scatter.png")  # Save scatter plot figure
#     plt.close()



import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm

# Set global font to Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# Define datasets and their corresponding time steps
# datasets = [
#     {"name": "Gaussian6_64", "path": "T:/MF_new/GrainAreaAvgGaussian6_64.mat", "time_step": 62},
#     {"name": "Gaussian_Square6_64", "path": "T:/MF_new/GrainAreaAvgGaussian_Square6_64.mat", "time_step": 49},
#     {"name": "Gaussian_Uniform6_64", "path": "T:/MF_new/GrainAreaAvgGaussian_Uniform6_64.mat", "time_step": 35},
# ]
#
# # Define the global range of sides (f) for consistent colormap
# global_sides = np.arange(1, 14)  # Assuming sides range from 1 to 13
# n_colors = len(global_sides)
#
# # Create a consistent colormap for all datasets
# colors = plt.cm.tab20(np.linspace(0, 1, n_colors))  # Generate discrete colors
# cmap = ListedColormap(colors)
# bounds = global_sides - 0.5
# bounds = np.append(bounds, global_sides[-1] + 0.5)  # Extend bounds to include max
# norm = BoundaryNorm(bounds, cmap.N)
#
# # Loop through each dataset
# for dataset in datasets:
#     # Load data
#     data_grain_area_avg = loadmat(dataset["path"])["GrainAreaAvg"]
#     data_grain_all_area = loadmat(dataset["path"].replace("GrainAreaAvg", "GrainAllArea"))["GrainAllArea"]
#     data_grain_all_sides = loadmat(dataset["path"].replace("GrainAreaAvg", "GrainAllSides"))["GrainAllSides"]
#     time_step = dataset["time_step"]
#
#     # Transpose matrices for easier handling
#     grain_area_avg = data_grain_area_avg.T
#     grain_all_area = data_grain_all_area.T
#     grain_all_sides = data_grain_all_sides.T
#
#     # Calculate slope from polyfit
#     pp = np.polyfit(np.arange(1, time_step * 2 + 1), grain_area_avg[:time_step * 2, 0], 1)
#     slope = pp[0]
#
#     # Process area data
#     a1 = grain_all_area[:, time_step].astype(float)
#     a2 = grain_all_area[:, time_step - 2].astype(float)
#     a3 = grain_all_area[:, time_step + 2].astype(float)
#
#     a_1 = a1[a1 > 0]
#     a_2 = a2[a1 > 0]
#     a_3 = a3[a1 > 0]
#
#     # dA_dt_norm = (a_3 - a_2) / 2/ (np.pi / 3) / slope
#     dA_dt_norm = (a_3 - a_2) / 4 / (np.pi / 3) / slope
#
#     # Process sides data
#     f1 = grain_all_sides[:, time_step].astype(float)
#     f = f1[a1 > 0]
#
#     # Calculate averages and std deviations for plotting
#     favg = []
#     dA_dt_normavg = []
#     dA_dt_normstd = []
#
#     for fi in global_sides:
#         findex = np.where(f == fi)[0]
#         if findex.size > 0:
#             favg.append(fi)
#             dA_dt_normavg.append(np.mean(dA_dt_norm[findex]))
#             dA_dt_normstd.append(np.std(dA_dt_norm[findex]))
#
#     favg = np.array(favg)
#     dA_dt_normavg = np.array(dA_dt_normavg)
#     dA_dt_normstd = np.array(dA_dt_normstd)
#
#     # Plot error bars
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.errorbar(favg, dA_dt_normavg, yerr=dA_dt_normstd, fmt='-o', markersize=6, linewidth=2, capsize=3)
#     ax.plot(favg, -(6 - favg), 'k-', linewidth=2)
#     ax.set_ylim(-10, 10)
#     ax.set_xticks([6])
#     ax.set_xlabel('Number of sides F', fontsize=30, labelpad=3)
#     ax.set_ylabel(r'$\frac{dA}{dt}/\left(\frac{\pi}{3} \mu \sigma\right)$', fontsize=30, labelpad=3)
#     ax.tick_params(axis='both', labelsize=30)
#     ax.axhline(0, color='k', linestyle='--', linewidth=1.5)
#     ax.axvline(6, color='k', linestyle='--', linewidth=1.5)
#
#     plt.tight_layout()
#     plt.savefig(f"{dataset['name']}_errorbar.png",bbox_inches='tight')  # Save error bar figure
#     plt.close()
#
#     # Second plot for normalized radius
#     r_norm = np.sqrt(a_1) / np.pi / np.mean(np.sqrt(a_1) / np.pi)
#
#     # Plot scatter with consistent colormap
#     fig, ax = plt.subplots(figsize=(8, 6))
#     scatter = ax.scatter(r_norm, dA_dt_norm, c=f, cmap=cmap, norm=norm, s=3)  # Smaller scatter size
#
#     # Add colorbar with mapping to unique sides
#     cbar = plt.colorbar(scatter, ax=ax, boundaries=bounds, ticks=global_sides, pad=0.005)
#     cbar.set_label('Number of sides F', fontsize=30)
#     cbar.ax.tick_params(labelsize=30)
#
#     cbar_ax = cbar.ax
#     cbar_pos = cbar_ax.get_position()
#     cbar_ax.set_position([cbar_pos.x0-0.1, cbar_pos.y0, 0.001, cbar_pos.height * 0.6])
#
#     ax.set_ylim(-10, 10)
#
#     # Set plot labels
#     ax.set_xlabel(r'Normalized radius $R/\langle R \rangle$', fontsize=30, labelpad=3)
#     ax.set_ylabel(r'$\frac{dA}{dt}/\left(\frac{\pi}{3} \mu \sigma\right)$', fontsize=30, labelpad=3)
#     ax.tick_params(axis='both', labelsize=30)
#
#     # plt.tight_layout()
#     plt.savefig(f"{dataset['name']}_scatter.png",bbox_inches='tight')  # Save scatter plot figure
#     plt.close()











# datasets = [
#     {"name": "MCP_primme_0.66", "path": "T:/MF_new/GrainAreaAvg_MCP_primme_0.66.mat", "time_step": 62},
#     {"name": "Gaussian12", "path": "T:/MF_new/GrainAreaAvgGaussian12.mat", "time_step": 28},
#     {"name": "Gaussian_Square12", "path": "T:/MF_new/GrainAreaAvgGaussian_Square12.mat", "time_step": 21},
#     {"name": "Gaussian_Uniform12", "path": "T:/MF_new/GrainAreaAvgGaussian_Uniform12.mat", "time_step": 15}
# ]
#
# # Define the global range of sides (f) for consistent colormap
# global_sides = np.arange(3, 12)  # Assuming sides range from 1 to 13
# n_colors = len(global_sides)
#
# # Create a consistent colormap for all datasets
# # colors = plt.cm.tab20(np.linspace(0, 1, n_colors))  # Generate discrete colors
# # cmap = ListedColormap(colors)
#
# # manual_colors = [
# #     "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
# #     "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#f45b5b", "#6c3483","#ffbf00"
# # ]
# # cmap = ListedColormap(manual_colors)
#
# colors = plt.cm.RdYlBu(np.linspace(0, 1, 9))
# cmap = ListedColormap(colors)
#
# bounds = global_sides - 0.5
# bounds = np.append(bounds, global_sides[-1] + 0.5)  # Extend bounds to include max
# norm = BoundaryNorm(bounds, cmap.N)
#
# # Loop through each dataset
# for dataset in datasets:
#     # Load data
#     data_grain_area_avg = loadmat(dataset["path"])["GrainAreaAvg"]
#     data_grain_all_area = loadmat(dataset["path"].replace("GrainAreaAvg", "GrainAllArea"))["GrainAllArea"]
#     data_grain_all_sides = loadmat(dataset["path"].replace("GrainAreaAvg", "GrainAllSides"))["GrainAllSides"]
#     time_step = dataset["time_step"]
#
#     # Transpose matrices for easier handling
#     grain_area_avg = data_grain_area_avg.T
#     grain_all_area = data_grain_all_area.T
#     grain_all_sides = data_grain_all_sides.T
#
#     # Calculate slope from polyfit
#     pp = np.polyfit(np.arange(1, time_step * 2 + 1), grain_area_avg[:time_step * 2, 0], 1)
#     slope = pp[0]
#
#     # Process area data
#     a1 = grain_all_area[:, time_step].astype(float)
#     a2 = grain_all_area[:, time_step - 1].astype(float)
#     a3 = grain_all_area[:, time_step + 1].astype(float)
#
#
#     a_1 = a1[a1 > 0]
#     a_2 = a2[a1 > 0]
#     a_3 = a3[a1 > 0]
#
#     dA_dt_norm = (a_3 - a_2) / 2/ (np.pi / 3) / slope
#
#     # Process sides data
#     f1 = grain_all_sides[:, time_step].astype(float)
#     f = f1[a1 > 0]
#
#     # Calculate averages and std deviations for plotting
#     favg = []
#     dA_dt_normavg = []
#     dA_dt_normstd = []
#
#     for fi in global_sides:
#         findex = np.where(f == fi)[0]
#         if findex.size > 0:
#             favg.append(fi)
#             dA_dt_normavg.append(np.mean(dA_dt_norm[findex]))
#             dA_dt_normstd.append(np.std(dA_dt_norm[findex]))
#
#     favg = np.array(favg)
#     dA_dt_normavg = np.array(dA_dt_normavg)
#     dA_dt_normstd = np.array(dA_dt_normstd)
#
#     # Plot error bars
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.errorbar(favg, dA_dt_normavg, yerr=dA_dt_normstd, fmt='-o', markersize=6, linewidth=2, capsize=3)
#     ax.plot(favg, -(6 - favg), 'k-', linewidth=2)
#     ax.set_ylim(-10, 10)
#     ax.set_xticks([6])
#     ax.set_xlabel('Number of sides F', fontsize=30, labelpad=3)
#     ax.set_ylabel(r'$dA/dt$', fontsize=30, labelpad=3)
#     ax.tick_params(axis='both', labelsize=30)
#     ax.yaxis.set_label_coords(-0.1, 0.5)
#     ax.axhline(0, color='k', linestyle='--', linewidth=1.5)
#     ax.axvline(6, color='k', linestyle='--', linewidth=1.5)
#
#     # plt.tight_layout()
#     plt.savefig(f"{dataset['name']}_errorbar.png",bbox_inches='tight')  # Save error bar figure
#     plt.close()
#
#     # Second plot for normalized radius
#     r_norm = np.sqrt(a_1) / np.pi / np.mean(np.sqrt(a_1) / np.pi)
#
#     # Plot scatter with consistent colormap
#     fig, ax = plt.subplots(figsize=(8, 6))
#     scatter = ax.scatter(r_norm, dA_dt_norm, c=f, cmap=cmap, norm=norm, s=3)  # Smaller scatter size
#
#     # Add colorbar with mapping to unique sides
#     cbar = plt.colorbar(scatter, ax=ax, boundaries=bounds, ticks=global_sides, pad=0)
#     # cbar.set_label('F', fontsize=30)
#     cbar.ax.tick_params(labelsize=30)
#
#     cbar_ax = cbar.ax
#     cbar_pos = cbar_ax.get_position()
#     cbar_ax.set_position([cbar_pos.x0-0.02, cbar_pos.y0, 0.03, cbar_pos.height])
#
#     ax.set_ylim(-10, 10)
#     # Set plot labels
#     ax.set_xlabel(r'$R/\langle R \rangle$', fontsize=30, labelpad=3)
#     # ax.set_ylabel(r'$\frac{dA}{dt}/\left(\frac{\pi}{3} \mu \sigma\right)$', fontsize=30, labelpad=3)
#     ax.set_ylabel(r'$dA/dt$', fontsize=30, labelpad=0)
#     ax.tick_params(axis='both', labelsize=30)
#     ax.yaxis.set_label_coords(-0.1, 0.5)
#
#     # plt.tight_layout()
#     plt.savefig(f"{dataset['name']}_scatter.png",bbox_inches='tight')  # Save scatter plot figure
#     plt.close()



import matplotlib
matplotlib.use('TkAgg')  # or 'QtAgg'
datasets = [
    # {"name": "Gaussian6_64", "path": "T:/MF_new/GrainAreaAvgGaussian6_64.mat", "time_step": 62},
    {"name": "Gaussian_Square6_64", "path": "T:/MF_new/GrainAreaAvgGaussian_Square6_64.mat", "time_step": 49},
    {"name": "Gaussian_Uniform6_64", "path": "T:/MF_new/GrainAreaAvgGaussian_Uniform6_64.mat", "time_step": 35},
]
# Define the global range of sides (f) for consistent colormap
global_sides = np.arange(3, 12)  # Assuming sides range from 1 to 13
n_colors = len(global_sides)

colors = plt.cm.RdYlBu(np.linspace(0, 1, 9))
cmap = ListedColormap(colors)



bounds = global_sides - 0.5
bounds = np.append(bounds, global_sides[-1] + 0.5)  # Extend bounds to include max
norm = BoundaryNorm(bounds, cmap.N)

# Loop through each dataset
for dataset in datasets:
    # Load data
    data_grain_area_avg = loadmat(dataset["path"])["GrainAreaAvg"]
    data_grain_all_area = loadmat(dataset["path"].replace("GrainAreaAvg", "GrainAllArea"))["GrainAllArea"]
    data_grain_all_sides = loadmat(dataset["path"].replace("GrainAreaAvg", "GrainAllSides"))["GrainAllSides"]
    time_step = dataset["time_step"]

    # Transpose matrices for easier handling
    grain_area_avg = data_grain_area_avg.T
    grain_all_area = data_grain_all_area.T
    grain_all_sides = data_grain_all_sides.T

    # Calculate slope from polyfit
    pp = np.polyfit(np.arange(1, time_step * 2 + 1), grain_area_avg[:time_step * 2, 0], 1)
    slope = pp[0]

    # Process area data
    a1 = grain_all_area[:, time_step].astype(float)
    a2 = grain_all_area[:, time_step - 1].astype(float)
    a3 = grain_all_area[:, time_step + 1].astype(float)


    a_1 = a1[a1 > 0]
    a_2 = a2[a1 > 0]
    a_3 = a3[a1 > 0]

    dA_dt_norm = (a_3 - a_2) / 2/ (np.pi / 3) / slope

    # Process sides data
    f1 = grain_all_sides[:, time_step].astype(float)
    f = f1[a1 > 0]

    # Calculate averages and std deviations for plotting
    favg = []
    dA_dt_normavg = []
    dA_dt_normstd = []

    for fi in global_sides:
        findex = np.where(f == fi)[0]
        if findex.size > 0:
            favg.append(fi)
            dA_dt_normavg.append(np.mean(dA_dt_norm[findex]))
            dA_dt_normstd.append(np.std(dA_dt_norm[findex]))

    favg = np.array(favg)
    dA_dt_normavg = np.array(dA_dt_normavg)
    dA_dt_normstd = np.array(dA_dt_normstd)

    # Plot error bars
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(favg, dA_dt_normavg, yerr=dA_dt_normstd, fmt='-o', markersize=6, linewidth=2, capsize=3)
    ax.plot(favg, -(6 - favg), 'k-', linewidth=2)
    ax.set_ylim(-10, 10)
    ax.set_xticks([6])
    ax.set_xlabel('Number of sides F', fontsize=30, labelpad=3)
    ax.set_ylabel(r'$dA/dt$', fontsize=30, labelpad=3)
    ax.tick_params(axis='both', labelsize=30)
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.axhline(0, color='k', linestyle='--', linewidth=1.5)
    ax.axvline(6, color='k', linestyle='--', linewidth=1.5)

    # plt.tight_layout()
    plt.savefig(f"{dataset['name']}_errorbar.png",bbox_inches='tight')  # Save error bar figure
    plt.close()

    # Second plot for normalized radius
    r_norm = np.sqrt(a_1) / np.pi / np.mean(np.sqrt(a_1) / np.pi)

    # Plot scatter with consistent colormap
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(r_norm, dA_dt_norm, c=f, cmap=cmap, norm=norm, s=3)  # Smaller scatter size

    # Add colorbar with mapping to unique sides
    cbar = plt.colorbar(scatter, ax=ax, boundaries=bounds, ticks=global_sides, pad=0)
    # cbar.set_label('F', fontsize=30)
    cbar.ax.tick_params(labelsize=30)

    cbar_ax = cbar.ax
    cbar_pos = cbar_ax.get_position()
    cbar_ax.set_position([cbar_pos.x0-0.02, cbar_pos.y0, 0.03, cbar_pos.height])

    ax.set_ylim(-10, 10)
    # Set plot labels
    ax.set_xlabel(r'$R/\langle R \rangle$', fontsize=30, labelpad=3)
    # ax.set_ylabel(r'$\frac{dA}{dt}/\left(\frac{\pi}{3} \mu \sigma\right)$', fontsize=30, labelpad=3)
    ax.set_ylabel(r'$dA/dt$', fontsize=30, labelpad=0)
    ax.tick_params(axis='both', labelsize=30)
    ax.yaxis.set_label_coords(-0.1, 0.5)

    # plt.tight_layout()
    plt.savefig(f"{dataset['name']}_scatter.png",bbox_inches='tight')  # Save scatter plot figure
    plt.close()