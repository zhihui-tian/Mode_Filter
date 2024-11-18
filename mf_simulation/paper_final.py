import torch
import numpy as np
import os
from tqdm import tqdm
import h5py
import torch.nn.functional as F
import imageio
import matplotlib.pyplot as plt
from unfoldNd import unfoldNd
import matplotlib.colors as mcolors
import time
import functions as fs

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
# export TORCH_USE_CUDA_DSA=1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""polycrystal case"""
with open("/blue/joel.harley/zhihui.tian/MF_new/MF-main/Case4.init", 'r') as file:
    content = file.readlines()
all=[]
for i in range(5760000):
    all.append(int(content[i+3].split()[1]))
k=np.reshape(all,(2400,2400))
ic=k.T
ea=fs.init2euler("/blue/joel.harley/zhihui.tian/MF_new/MF-main/Case4.init")[0,:,:]


#################################################################################################################################################
# ims_id = fs.run_mf(ic, ea,i,'Gaussian', nsteps=2000, cut=0, cov=25,num_samples=32, memory_limit=1e10,device=device)
# ims_id = fs.run_mf(ic, ea,i,'Gaussian', nsteps=2000, cut=0, cov=25,num_samples=64, memory_limit=1e10,device=device)

# ims_id = fs.run_mf(ic, ea,i,'Gaussian_Uniform', nsteps=4000, cut=0, cov=25,num_samples=64, memory_limit=1e10,device=device)
# ims_id = fs.run_mf(ic, ea,i,'Gaussian_Square', nsteps=4000, cut=0, cov=25,num_samples=64, memory_limit=1e10,device=device)

# """4000steps of square rotation"""
# ims_id = fs.run_mf(ic, ea,30,'Gaussian_Square_Rotation', nsteps=4000, cut=0, cov=25,num_samples=64, memory_limit=1e10,device=device)
# ims_id = fs.run_mf(ic, ea,60,'Gaussian_Square_Rotation', nsteps=4000, cut=0, cov=25,num_samples=64, memory_limit=1e10,device=device)
# ims_id = fs.run_mf(ic, ea,90,'Gaussian_Square_Rotation', nsteps=4000, cut=0, cov=25,num_samples=64, memory_limit=1e10,device=device)

# """4000steps of uniform with rotation"""
# ims_id = fs.run_mf(ic, ea,30,'Gaussian_Uniform_Rotation', nsteps=4000, cut=0, cov=25,num_samples=64, memory_limit=1e10,device=device)
# ims_id = fs.run_mf(ic, ea,60,'Gaussian_Uniform_Rotation', nsteps=4000, cut=0, cov=25,num_samples=64, memory_limit=1e10,device=device)
# ims_id = fs.run_mf(ic, ea,90,'Gaussian_Uniform_Rotation', nsteps=4000, cut=0, cov=25,num_samples=64, memory_limit=1e10,device=device)


#################################################################################################################################################
"""9 sets for figures"""
# f=h5py.File("/home/zhihui.tian/blue_gator_group/zhihui.tian/MF_new/data/mf_sz(2400x2400)_ng(20000)_nsteps(2000)_cov(25)_numnei(64)_cut(0)_exp(Gaussian)_(5759999).h5",'r')
# np.save('Gaussian_2000step.npy',f['sim0']['ims_id'])
# f=h5py.File("/home/zhihui.tian/blue_gator_group/zhihui.tian/MF_new/data/mf_sz(2400x2400)_ng(20000)_nsteps(4000)_cov(25)_numnei(64)_cut(0)_exp(Gaussian_Square)_(5759999).h5",'r')
# np.save('Gaussian_Square_4000step.npy',f['sim0']['ims_id'])
# f=h5py.File("/home/zhihui.tian/blue_gator_group/zhihui.tian/MF_new/data/mf_sz(2400x2400)_ng(20000)_nsteps(4000)_cov(25)_numnei(64)_cut(0)_exp(Gaussian_Uniform)_(5759999).h5",'r')
# np.save('Gaussian_Uniform_4000step.npy',f['sim0']['ims_id'])

# f=h5py.File("/home/zhihui.tian/blue_gator_group/zhihui.tian/MF_new/data/mf_sz(2400x2400)_ng(20000)_nsteps(4000)_cov(25)_numnei(64)_cut(0)_exp(Gaussian_Square_Rotation)_(30).h5",'r')
# np.save('Gaussian_Square_4000step_30.npy',f['sim0']['ims_id'])
# f=h5py.File("/home/zhihui.tian/blue_gator_group/zhihui.tian/MF_new/data/mf_sz(2400x2400)_ng(20000)_nsteps(4000)_cov(25)_numnei(64)_cut(0)_exp(Gaussian_Square_Rotation)_(60).h5",'r')
# np.save('Gaussian_Square_4000step_60.npy',f['sim0']['ims_id'])
# f=h5py.File("/home/zhihui.tian/blue_gator_group/zhihui.tian/MF_new/data/mf_sz(2400x2400)_ng(20000)_nsteps(4000)_cov(25)_numnei(64)_cut(0)_exp(Gaussian_Square_Rotation)_(90).h5",'r')
# np.save('Gaussian_Square_4000step_90.npy',f['sim0']['ims_id'])

# f=h5py.File("/home/zhihui.tian/blue_gator_group/zhihui.tian/MF_new/data/mf_sz(2400x2400)_ng(20000)_nsteps(4000)_cov(25)_numnei(64)_cut(0)_exp(Gaussian_Uniform_Rotation)_(30).h5",'r')
# np.save('Gaussian_Uniform_4000step_30.npy',f['sim0']['ims_id'])
# f=h5py.File("/home/zhihui.tian/blue_gator_group/zhihui.tian/MF_new/data/mf_sz(2400x2400)_ng(20000)_nsteps(4000)_cov(25)_numnei(64)_cut(0)_exp(Gaussian_Uniform_Rotation)_(60).h5",'r')
# np.save('Gaussian_Uniform_4000step_60.npy',f['sim0']['ims_id'])
# f=h5py.File("/home/zhihui.tian/blue_gator_group/zhihui.tian/MF_new/data/mf_sz(2400x2400)_ng(20000)_nsteps(4000)_cov(25)_numnei(64)_cut(0)_exp(Gaussian_Uniform_Rotation)_(90).h5",'r')
# np.save('Gaussian_Uniform_4000step_90.npy',f['sim0']['ims_id'])







#############################################################################################################################
# def get_indices(type,steps):
#     num_grain = []
#     for i in range(steps):
#         num_grain.append(np.unique(type[i,0,:,:]).shape[0])
#     def find_nearest_indices(array, values):   ### find indices of gaussian square and uniform distribution
#         indices = []
#         for value in values:
#             if value in array:
#                 # Exact match: find the index
#                 index = np.where(array == value)[0][0]
#             else:
#                 # Find the index of the nearest value
#                 index = (np.abs(array - value)).argmin()
#             indices.append(index)
#         return indices
#     values = [20000,12590,8369,447,300]
#     indices_final = find_nearest_indices(np.array(num_grain), values)
#     return indices_final



# import h5py
# import numpy as np
# gaussian = np.load('Gaussian_2000step.npy')
# gaussian_indices = get_indices(gaussian,2000)
# np.save('Plot_gaussian_array.npy',gaussian[gaussian_indices,:,:,:])

# ############################### gaussian square

# gaussian_square = np.load('Gaussian_Square_4000step.npy')
# gaussian_square_indices = get_indices(gaussian_square,4000)
# np.save('Plot_gaussian_square_array.npy',gaussian_square[gaussian_square_indices,:,:,:])

# gaussian_square30 = np.load('Gaussian_Square_4000step_30.npy')
# gaussian_square30_indices = get_indices(gaussian_square30,4000)
# np.save('Plot_gaussian_square30_array.npy',gaussian_square30[gaussian_square30_indices,:,:,:])

# gaussian_square60 = np.load('Gaussian_Square_4000step_60.npy')
# gaussian_square60_indices = get_indices(gaussian_square60,4000)
# np.save('Plot_gaussian_square60_array.npy',gaussian_square60[gaussian_square60_indices,:,:,:])

# gaussian_square90 = np.load('Gaussian_Square_4000step_90.npy')
# gaussian_square90_indices = get_indices(gaussian_square90,4000)
# np.save('Plot_gaussian_square90_array.npy',gaussian_square90[gaussian_square90_indices,:,:,:])



# ############################# uniform




# gaussian_uniform = np.load('Gaussian_Uniform_4000step.npy')
# gaussian_uniform_indices = get_indices(gaussian_uniform,4000)
# np.save('Plot_gaussian_uniform_array.npy',gaussian_uniform[gaussian_uniform_indices,:,:,:])


# gaussian_uniform30 = np.load('Gaussian_Uniform_4000step_30.npy')
# gaussian_uniform30_indices = get_indices(gaussian_uniform30,4000)
# np.save('Plot_gaussian_uniform30_array.npy',gaussian_uniform30[gaussian_uniform30_indices,:,:,:])

# gaussian_uniform60 = np.load('Gaussian_Uniform_4000step_60.npy')
# gaussian_uniform60_indices = get_indices(gaussian_uniform60,4000)
# np.save('Plot_gaussian_uniform60_array.npy',gaussian_uniform60[gaussian_uniform60_indices,:,:,:])

# gaussian_uniform90 = np.load('Gaussian_Uniform_4000step_30.npy')
# gaussian_uniform90_indices = get_indices(gaussian_uniform90,4000)
# np.save('Plot_gaussian_uniform90_array.npy',gaussian_uniform90[gaussian_uniform90_indices,:,:,:])


###################################################################################################### vnmr
# fs.compute_grain_stats("/home/zhihui.tian/blue_gator_group/zhihui.tian/MF_new/data/mf_sz(2400x2400)_ng(20000)_nsteps(2000)_cov(25)_numnei(64)_cut(0)_exp(Gaussian)_(5759999).h5")
fs.compute_grain_stats("/home/zhihui.tian/blue_gator_group/zhihui.tian/MF_new/data/mf_sz(2400x2400)_ng(20000)_nsteps(4000)_cov(25)_numnei(64)_cut(0)_exp(Gaussian_Square)_(5759999).h5")
fs.compute_grain_stats("/home/zhihui.tian/blue_gator_group/zhihui.tian/MF_new/data/mf_sz(2400x2400)_ng(20000)_nsteps(4000)_cov(25)_numnei(64)_cut(0)_exp(Gaussian_Uniform)_(5759999).h5")





























































































"""get indices that has certain number of grain"""
# num_grain = []
# for i in range(4000):
#     num_grain.append(np.unique(gaussian_square[i,0,:,:]).shape[0])
# """find indices of gaussian square and uniform distribution"""
# def find_nearest_indices(array, values):   ### find indices of gaussian square and uniform distribution
#     indices = []
#     for value in values:
#         if value in array:
#             # Exact match: find the index
#             index = np.where(array == value)[0][0]
#         else:
#             # Find the index of the nearest value
#             index = (np.abs(array - value)).argmin()
#         indices.append(index)
#     return indices
# values = [20000,12590,8369,447,300]
# indices = find_nearest_indices(np.array(num_grain), values)


""""""""""""""""""""""""""""""""""""""""""""""""
"""4000steps of uniform with rotation"""
# ims_id = fs.run_mf(ic, ea,30,'Gaussian_Uniform_Rotation', nsteps=4000, cut=0, cov=25,num_samples=64, memory_limit=1e10,device=device)
# ims_id = fs.run_mf(ic, ea,60,'Gaussian_Uniform_Rotation', nsteps=4000, cut=0, cov=25,num_samples=64, memory_limit=1e10,device=device)
# ims_id = fs.run_mf(ic, ea,90,'Gaussian_Uniform_Rotation', nsteps=4000, cut=0, cov=25,num_samples=64, memory_limit=1e10,device=device)
# f=h5py.File("/home/zhihui.tian/blue_gator_group/zhihui.tian/MF_new/data/mf_sz(2400x2400)_ng(20000)_nsteps(4000)_cov(25)_numnei(64)_cut(0)_exp(Gaussian_Uniform_Rotation)_(30).h5",'r')
# np.save('Gaussian_Uniform_4000step_30.npy',f['sim0']['ims_id'])
# f=h5py.File("/home/zhihui.tian/blue_gator_group/zhihui.tian/MF_new/data/mf_sz(2400x2400)_ng(20000)_nsteps(4000)_cov(25)_numnei(64)_cut(0)_exp(Gaussian_Uniform_Rotation)_(60).h5",'r')
# np.save('Gaussian_Uniform_4000step_60.npy',f['sim0']['ims_id'])
# f=h5py.File("/home/zhihui.tian/blue_gator_group/zhihui.tian/MF_new/data/mf_sz(2400x2400)_ng(20000)_nsteps(4000)_cov(25)_numnei(64)_cut(0)_exp(Gaussian_Uniform_Rotation)_(90).h5",'r')
# np.save('Gaussian_Uniform_4000step_90.npy',f['sim0']['ims_id'])


# import h5py
# import numpy as np
# gaussian_uniform_30 = np.load('Gaussian_Uniform_4000step_30.npy')
# gaussian_uniform_60 = np.load('Gaussian_Uniform_4000step_60.npy')
# gaussian_uniform_90 = np.load('Gaussian_Uniform_4000step_90.npy')


# """get indices that has certain number of grain"""
# num_grain = []
# for i in range(4000):
#     num_grain.append(np.unique(gaussian_uniform_30[i,0,:,:]).shape[0])
# """find indices of gaussian square and uniform distribution"""
# def find_nearest_indices(array, values):   ### find indices of gaussian square and uniform distribution
#     indices = []
#     for value in values:
#         if value in array:
#             # Exact match: find the index
#             index = np.where(array == value)[0][0]
#         else:
#             # Find the index of the nearest value
#             index = (np.abs(array - value)).argmin()
#         indices.append(index)
#     return indices
# values = [20000,12590,8369,447,300]
# indices30 = find_nearest_indices(np.array(num_grain), values)
# np.save('uniform_4000_rotate30.npy',gaussian_uniform_60[indices30,:,:,:])

# num_grain = []
# for i in range(4000):
#     num_grain.append(np.unique(gaussian_uniform_60[i,0,:,:]).shape[0])
# """find indices of gaussian square and uniform distribution"""
# def find_nearest_indices(array, values):   ### find indices of gaussian square and uniform distribution
#     indices = []
#     for value in values:
#         if value in array:
#             # Exact match: find the index
#             index = np.where(array == value)[0][0]
#         else:
#             # Find the index of the nearest value
#             index = (np.abs(array - value)).argmin()
#         indices.append(index)
#     return indices
# values = [20000,12590,8369,447,300]
# indices60 = find_nearest_indices(np.array(num_grain), values)
# np.save('uniform_4000_rotate60.npy',gaussian_uniform_60[indices60,:,:,:])

# num_grain = []
# for i in range(4000):
#     num_grain.append(np.unique(gaussian_uniform_90[i,0,:,:]).shape[0])
# """find indices of gaussian square and uniform distribution"""
# def find_nearest_indices(array, values):   ### find indices of gaussian square and uniform distribution
#     indices = []
#     for value in values:
#         if value in array:
#             # Exact match: find the index
#             index = np.where(array == value)[0][0]
#         else:
#             # Find the index of the nearest value
#             index = (np.abs(array - value)).argmin()
#         indices.append(index)
#     return indices
# values = [20000,12590,8369,447,300]
# indices90 = find_nearest_indices(np.array(num_grain), values)
# np.save('uniform_4000_rotate90.npy',gaussian_uniform_90[indices90,:,:,:])




"""4000steps of square"""
# ims_id = fs.run_mf(ic, ea,30,'Gaussian_Square_Rotation', nsteps=4000, cut=0, cov=25,num_samples=64, memory_limit=1e10,device=device)
# ims_id = fs.run_mf(ic, ea,60,'Gaussian_Square_Rotation', nsteps=4000, cut=0, cov=25,num_samples=64, memory_limit=1e10,device=device)
# ims_id = fs.run_mf(ic, ea,90,'Gaussian_Square_Rotation', nsteps=4000, cut=0, cov=25,num_samples=64, memory_limit=1e10,device=device)
# f=h5py.File("/home/zhihui.tian/blue_gator_group/zhihui.tian/MF_new/data/mf_sz(2400x2400)_ng(20000)_nsteps(4000)_cov(25)_numnei(64)_cut(0)_exp(Gaussian_Square_Rotation)_(30).h5",'r')
# np.save('Gaussian_Square_4000step_30.npy',f['sim0']['ims_id'])
# f=h5py.File("/home/zhihui.tian/blue_gator_group/zhihui.tian/MF_new/data/mf_sz(2400x2400)_ng(20000)_nsteps(4000)_cov(25)_numnei(64)_cut(0)_exp(Gaussian_Square_Rotation)_(60).h5",'r')
# np.save('Gaussian_Square_4000step_60.npy',f['sim0']['ims_id'])
# f=h5py.File("/home/zhihui.tian/blue_gator_group/zhihui.tian/MF_new/data/mf_sz(2400x2400)_ng(20000)_nsteps(4000)_cov(25)_numnei(64)_cut(0)_exp(Gaussian_Square_Rotation)_(90).h5",'r')
# np.save('Gaussian_Square_4000step_90.npy',f['sim0']['ims_id'])


# import h5py
# import numpy as np
# gaussian_Square_30 = np.load('Gaussian_Square_4000step_30.npy')
# gaussian_Square_60 = np.load('Gaussian_Square_4000step_60.npy')
# gaussian_Square_90 = np.load('Gaussian_Square_4000step_90.npy')


# """get indices that has certain number of grain"""
# num_grain = []
# for i in range(4000):
#     num_grain.append(np.unique(gaussian_Square_30[i,0,:,:]).shape[0])
# """find indices of gaussian square and uniform distribution"""
# def find_nearest_indices(array, values):   ### find indices of gaussian square and uniform distribution
#     indices = []
#     for value in values:
#         if value in array:
#             # Exact match: find the index
#             index = np.where(array == value)[0][0]
#         else:
#             # Find the index of the nearest value
#             index = (np.abs(array - value)).argmin()
#         indices.append(index)
#     return indices
# values = [20000,12590,8369,447,300]
# indices30 = find_nearest_indices(np.array(num_grain), values)
# np.save('gaussian_Square_rotate30.npy',gaussian_Square_30[indices30,:,:,:])

# num_grain = []
# for i in range(4000):
#     num_grain.append(np.unique(gaussian_Square_60[i,0,:,:]).shape[0])
# """find indices of gaussian square and uniform distribution"""
# def find_nearest_indices(array, values):   ### find indices of gaussian square and uniform distribution
#     indices = []
#     for value in values:
#         if value in array:
#             # Exact match: find the index
#             index = np.where(array == value)[0][0]
#         else:
#             # Find the index of the nearest value
#             index = (np.abs(array - value)).argmin()
#         indices.append(index)
#     return indices
# values = [20000,12590,8369,447,300]
# indices60 = find_nearest_indices(np.array(num_grain), values)
# np.save('gaussian_Square_rotate60.npy',gaussian_Square_60[indices60,:,:,:])

# num_grain = []
# for i in range(4000):
#     num_grain.append(np.unique(gaussian_Square_90[i,0,:,:]).shape[0])
# """find indices of gaussian square and uniform distribution"""
# def find_nearest_indices(array, values):   ### find indices of gaussian square and uniform distribution
#     indices = []
#     for value in values:
#         if value in array:
#             # Exact match: find the index
#             index = np.where(array == value)[0][0]
#         else:
#             # Find the index of the nearest value
#             index = (np.abs(array - value)).argmin()
#         indices.append(index)
#     return indices
# values = [20000,12590,8369,447,300]
# indices90 = find_nearest_indices(np.array(num_grain), values)
# np.save('gaussian_Square_rotate90.npy',gaussian_Square_90[indices90,:,:,:])


