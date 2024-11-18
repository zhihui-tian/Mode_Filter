#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:33:57 2023

@author: Lin
"""

import os

current_path = os.getcwd()
import numpy as np
from numpy import seterr

seterr(all='raise')
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm
import sys

import myInput
import PACKAGE_MP_Linear as linear2d
import h5py


def simple_magnitude(freqArray):
    xLim = [0, 360]
    binValue = 10.01
    binNum = round((abs(xLim[0]) + abs(xLim[1])) / binValue)
    xCor = np.linspace((xLim[0] + binValue / 2), (xLim[1] - binValue / 2), binNum)

    freqArray_circle = np.ones(binNum)
    freqArray_circle = freqArray_circle / sum(freqArray_circle * binValue)

    magnitude_max = np.max(abs(freqArray - freqArray_circle)) / np.average(freqArray_circle)
    magnitude_ave = np.average(abs(freqArray - freqArray_circle)) / np.average(freqArray_circle)

    magnitude_stan = np.sqrt(
        np.sum((abs(freqArray - freqArray_circle) / np.average(freqArray_circle) - magnitude_ave) ** 2) / binNum)

    return magnitude_ave, magnitude_stan

    # coeff_high = abs(np.cos((xCor-90)/180*np.pi))
    # coeff_low = abs(np.cos((xCor)/180*np.pi))
    # return np.sum(freqArray * coeff_high)/np.sum(freqArray * coeff_low)

def simple_magnitude_first_dimen(freqArray):
    xLim = [0, 90]
    binValue = 10.01
    binNum = round((abs(xLim[0]) + abs(xLim[1])) / binValue)
    xCor = np.linspace((xLim[0] + binValue / 2), (xLim[1] - binValue / 2), binNum)

    freqArray_circle = np.ones(binNum)
    freqArray_circle = freqArray_circle / sum(freqArray_circle * binValue)

    magnitude_max = np.max(abs(freqArray - freqArray_circle)) / np.average(freqArray_circle)
    magnitude_ave = np.average(abs(freqArray - freqArray_circle)) / np.average(freqArray_circle)

    magnitude_stan = np.sqrt(
        np.sum((abs(freqArray - freqArray_circle) / np.average(freqArray_circle) - magnitude_ave) ** 2) / binNum)

    return magnitude_ave, magnitude_stan

def simple_magnitude_abs(freqArray):
    # xLim = [0, 90]
    # binValue = 10.01
    # binNum = round((abs(xLim[0]) + abs(xLim[1])) / binValue)
    # xCor = np.linspace((xLim[0] + binValue / 2), (xLim[1] - binValue / 2), binNum)
    # freqArray_circle = np.ones(binNum)
    # # freqArray_circle = freqArray_circle / sum(freqArray_circle * binValue)
    # freqArray_circle = freqArray_circle / sum(freqArray_circle * 1.01)
    # freqArray *= 40
    # magnitude_ave = np.sum(abs(freqArray - freqArray_circle))

    # xLim = [0, 360]
    # binValue = 1.01
    # binNum = round((abs(xLim[0]) + abs(xLim[1])) / binValue)
    # xCor = np.linspace((xLim[0] + binValue / 2), (xLim[1] - binValue / 2), binNum)


    xLim = [0, 360]
    binNum = 360
    start = xLim[0] + 0.5
    end = 360
    binValue = (end - start) / (binNum - 1)
    xCor = np.linspace(start, end - binValue, binNum)


    freqArray_circle = np.ones(binNum)
    # freqArray_circle = freqArray_circle / sum(freqArray_circle * binValue)
    freqArray_circle = freqArray_circle / sum(freqArray_circle * 1.01)

    # magnitude_ave = np.sum(abs(freqArray - freqArray_circle))
    magnitude_ave = np.average(abs(freqArray - freqArray_circle)) / np.average(freqArray_circle)

    return magnitude_ave


def fit_ellipse_for_poly(micro_matrix, sites_list, step):  # failure

    # For circle, we only care about the circular grain
    grains_num = len(sites_list)
    # grains_num_real = np.sum([len(sites_list[i])>0 for i in range(len(sites_list))])
    sites_num_list = np.zeros(grains_num)
    # Calculate the area
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            grain_id = int(micro_matrix[step, i, j, 0] - 1)
            sites_num_list[grain_id] += 1
    center_list, _ = get_poly_center(micro_matrix, step)

    a_square_list = np.ones(grains_num)
    b_square_list = np.ones(grains_num)
    unphysic_result = 0
    grains_num_real = 0.001
    for i in range(grains_num):
        array = np.array(sites_list[i])
        grain_center = center_list[i]

        # Avoid the really small grains
        rest_site_num = 10
        if len(array) < rest_site_num or (center_list[i, 0] < 0.1 and center_list[i, 1] < 0.1):
            a_square_list[i] = 1
            b_square_list[i] = 1
            continue

        my_list = []
        prefered_angles = np.linspace(0, 2 * np.pi, rest_site_num + 1)[:rest_site_num]
        max_angles = np.ones(rest_site_num) * 2 * np.pi
        predered_sites = np.zeros((rest_site_num, 2))
        for n in range(len(array)):
            current_site_angle = math.atan2(array[n, 0] - grain_center[0], array[n, 1] - grain_center[1]) + np.pi
            my_list.append(current_site_angle)
            min_angle = np.min(abs(prefered_angles - current_site_angle))
            min_angle_index = np.argmin(abs(prefered_angles - current_site_angle))
            if min_angle < max_angles[min_angle_index]:
                max_angles[min_angle_index] = min_angle
                predered_sites[min_angle_index] = array[n]

        array = predered_sites
        grains_num_real += 1
        # Get the self-variable
        X = array[:, 0]
        Y = array[:, 1]

        # Calculation Kernel
        K_mat = np.array([X ** 2, X * Y, Y ** 2, X, Y]).T
        Y_mat = -np.ones_like(X)
        X_mat = np.linalg.lstsq(K_mat, Y_mat, rcond=None)[0].squeeze()

        # Calculate the long and short axis
        center_base = 4 * X_mat[0] * X_mat[2] - X_mat[1] * X_mat[1]
        center_x = (X_mat[1] * X_mat[4] - 2 * X_mat[2] * X_mat[3]) / center_base
        center_y = (X_mat[1] * X_mat[3] - 2 * X_mat[0] * X_mat[4]) / center_base
        axis_square_root = np.sqrt((X_mat[0] - X_mat[2]) ** 2 + X_mat[1] ** 2)
        a_square = 2 * (X_mat[0] * center_x * center_x + X_mat[2] * center_y * center_y + X_mat[
            1] * center_x * center_y - 1) / (X_mat[0] + X_mat[2] + axis_square_root)
        b_square = 2 * (X_mat[0] * center_x * center_x + X_mat[2] * center_y * center_y + X_mat[
            1] * center_x * center_y - 1) / (X_mat[0] + X_mat[2] - axis_square_root)

        #  Avoid the grains with strange shape
        if a_square < 0 or b_square < 0:
            # matrix = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
            # for s in range(len(array)):
            #     matrix[int(array[s,0]),int(array[s,1])] = 1
            # plt.close()
            # plt.imshow(matrix)
            a_square_list[i] = 1
            b_square_list[i] = 1
            unphysic_result += 1  # sites_num_list[i]
            continue
        # print(f"a: {np.sqrt(a_square)}, b: {np.sqrt(b_square)}")
        a_square_list[i] = a_square
        b_square_list[i] = b_square
    print(f"The unphysical result is {round(unphysic_result / grains_num_real * 100, 3)}%")

    return np.sum(b_square_list * sites_num_list) / np.sum(a_square_list * sites_num_list)


def get_poly_center(micro_matrix, step):
    # Get the center of all non-periodic grains in matrix
    num_grains = int(np.max(micro_matrix[step, :]))
    center_list = np.zeros((num_grains, 2))
    sites_num_list = np.zeros(num_grains)
    ave_radius_list = np.zeros(num_grains)
    coord_refer_i = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    coord_refer_j = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            coord_refer_i[i, j] = i
            coord_refer_j[i, j] = j

    table = micro_matrix[step, :, :, 0]
    for i in range(num_grains):
        sites_num_list[i] = np.sum(table == i + 1)

        if (sites_num_list[i] < 5) or \
                (np.max(coord_refer_i[table == i + 1]) - np.min(coord_refer_i[table == i + 1]) == micro_matrix.shape[
                    1]) or \
                (np.max(coord_refer_j[table == i + 1]) - np.min(coord_refer_j[table == i + 1]) == micro_matrix.shape[
                    2]):  # grains on bc are ignored
            center_list[i, 0] = 0
            center_list[i, 1] = 0
            sites_num_list[i] = 0
        else:
            center_list[i, 0] = (np.max(coord_refer_i[table == i + 1]) + np.min(coord_refer_i[table == i + 1])) / 2
            center_list[i, 1] = (np.max(coord_refer_j[table == i + 1]) + np.min(coord_refer_j[table == i + 1])) / 2
    ave_radius_list = np.sqrt(sites_num_list / np.pi)
    # print(np.max(sites_num_list))

    return center_list, ave_radius_list


def get_poly_statistical_radius(micro_matrix, sites_list, step):
    # Get the max offset of average radius and real radius
    center_list, ave_radius_list = get_poly_center(micro_matrix, step)
    num_grains = int(np.max(micro_matrix[step, :]))

    max_radius_offset_list = np.zeros(num_grains)
    for n in range(num_grains):
        center = center_list[n]
        ave_radius = ave_radius_list[n]
        sites = sites_list[n]

        if ave_radius != 0:
            for sitei in sites:
                [i, j] = sitei
                current_radius = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
                radius_offset = abs(current_radius - ave_radius)
                if radius_offset > max_radius_offset_list[n]: max_radius_offset_list[n] = radius_offset

            max_radius_offset_list[n] = max_radius_offset_list[n] / ave_radius

    # max_radius_offset = np.average(max_radius_offset_list[max_radius_offset_list!=0])
    area_list = np.pi * ave_radius_list * ave_radius_list
    if np.sum(area_list) == 0:
        max_radius_offset = 0
    else:
        max_radius_offset = np.sum(max_radius_offset_list * area_list) / np.sum(area_list)

    return max_radius_offset


def get_poly_statistical_ar(micro_matrix, step):
    # Get the average aspect ratio
    num_grains = int(np.max(micro_matrix[step, :]))
    sites_num_list = np.zeros(num_grains)
    coord_refer_i = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    coord_refer_j = np.zeros((micro_matrix.shape[1], micro_matrix.shape[2]))
    for i in range(micro_matrix.shape[1]):
        for j in range(micro_matrix.shape[2]):
            coord_refer_i[i, j] = i
            coord_refer_j[i, j] = j

    aspect_ratio_i = np.zeros((num_grains, 2))
    aspect_ratio_j = np.zeros((num_grains, 2))
    aspect_ratio = np.zeros(num_grains)
    table = micro_matrix[step, :, :, 0]
    for i in range(num_grains):
        sites_num_list[i] = np.sum(table == i + 1)
        aspect_ratio_i[i, 0] = len(list(set(coord_refer_i[table == i + 1])))
        aspect_ratio_j[i, 1] = len(list(set(coord_refer_j[table == i + 1])))
        if aspect_ratio_j[i, 1] == 0:
            aspect_ratio[i] = 0
        else:
            aspect_ratio[i] = aspect_ratio_i[i, 0] / aspect_ratio_j[i, 1]

    # aspect_ratio = np.average(aspect_ratio[aspect_ratio!=0])
    aspect_ratio = np.sum(aspect_ratio * sites_num_list) / np.sum(sites_num_list)

    return aspect_ratio


def get_normal_vector(grain_structure_figure_one, grain_num):
    nx = grain_structure_figure_one.shape[0]
    ny = grain_structure_figure_one.shape[1]
    ng = np.max(grain_structure_figure_one)
    cores = 8
    loop_times = 5
    P0 = grain_structure_figure_one
    R = np.zeros((nx, ny, 2))
    smooth_class = linear2d.linear_class(nx, ny, ng, cores, loop_times, P0, R)

    smooth_class.linear_main("inclination")
    P = smooth_class.get_P()
    # sites = smooth_class.get_gb_list(1)
    # print(len(sites))
    # for id in range(2,grain_num+1): sites += smooth_class.get_gb_list(id)
    # print(len(sites))
    sites = smooth_class.get_all_gb_list()
    sites_together = []
    for id in range(len(sites)): sites_together += sites[id]
    print("Total num of GB sites: " + str(len(sites_together)))

    return P, sites_together, sites


def get_normal_vector_slope(P, sites, step, para_name, bias=None):
    # xLim = [0, 360]
    # binValue = 10.01
    # binNum = round((abs(xLim[0]) + abs(xLim[1])) / binValue)
    # xCor = np.linspace((xLim[0] + binValue / 2), (xLim[1] - binValue / 2), binNum)
    #
    # freqArray = np.zeros(binNum)
    # degree = []

    xLim = [0, 360]
    binNum = 360

    start = xLim[0] + 0.5
    end = 360
    binValue = (end - start) / (binNum - 1)
    xCor = np.linspace(start, end - binValue, binNum)

    freqArray = np.zeros(binNum)
    degree = []

    for sitei in sites:
        [i, j] = sitei
        dx, dy = myInput.get_grad(P, i, j)
        degree.append(math.atan2(-dy, dx) + math.pi)
        # if dx == 0:
        #     degree.append(math.pi/2)
        # elif dy >= 0:
        #     degree.append(abs(math.atan(-dy/dx)))
        # elif dy < 0:
        #     degree.append(abs(math.atan(dy/dx)))
    for i in range(len(degree)):
        freqArray[int((degree[i] / math.pi * 180 - xLim[0]) / binValue)] += 1
    freqArray = freqArray / sum(freqArray * binValue)

    if bias is not None:
        freqArray = freqArray + bias
        freqArray = freqArray / sum(freqArray * binValue)
    # Plot
    # plt.close()
    # fig = plt.figure(figsize=(5, 5))
    # ax = plt.gca(projection='polar')

    # ax.set_thetagrids(np.arange(0.0, 360.0, 20.0),fontsize=14)
    # ax.set_thetamin(0.0)
    # ax.set_thetamax(360.0)

    # ax.set_rgrids(np.arange(0, 0.008, 0.004))
    # ax.set_rlabel_position(0.0)  # 标签显示在0°
    # ax.set_rlim(0.0, 0.008)  # 标签范围为[0, 5000)
    # ax.set_yticklabels(['0', '0.004'],fontsize=14)

    # ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    # ax.set_axisbelow('True')

    """CALCULATE COVARIANCE"""
    # x_dir =np.append(xCor, xCor[0]) / 180 * math.pi
    # y_dir =np.append(freqArray, freqArray[0])
    # cov_val = np.cov(x_dir, y_dir)[0, 1]

    # label_text = r"$\step={:.2f}$, $\mu={:.8f}$".format(step, cov_val)
    # label_text = "(step_{}, cov_{:.8f})".format(step, cov_val)
    # plt.plot(np.append(xCor, xCor[0]) / 180 * math.pi, np.append(freqArray, freqArray[0]), linewidth=2,
    #          label=label_text)
    plt.plot(np.append(xCor, xCor[0]) / 180 * math.pi, np.append(freqArray, freqArray[0]), linewidth=2,label=para_name)

    # fitting
    fit_coeff = np.polyfit(xCor, freqArray, 1)
    return freqArray

###########################################################################################figure2
# npy_file_aniso_mcp = np.load(r"T:\MF_new\prepare_paper_figure\figure3plot\mcp_plot.npy")
# npy_file_aniso_gaussian = np.load(r"T:\MF_new\Plot_gaussian_array.npy")
# npy_file_aniso_gaussian_square = np.load(r"T:\MF_new\Plot_gaussian_square_array.npy")
# npy_file_aniso_uniform = np.load(r"T:\MF_new\Plot_gaussian_uniform_array.npy")
#
# plt.imshow(npy_file_aniso_mcp[-1,0,:,:])
# plt.savefig('figure2_mcp.png', format='png', dpi=300, bbox_inches='tight')
# plt.imshow(npy_file_aniso_gaussian[-1,0,:,:])
# plt.savefig('figure2_gaussian.png', format='png', dpi=300, bbox_inches='tight')
# plt.imshow(npy_file_aniso_gaussian_square[-1,0,:,:])
# plt.savefig('figure2_gaussian_square.png', format='png', dpi=300, bbox_inches='tight')
# plt.imshow(npy_file_aniso_uniform[-1,0,:,:])
# plt.savefig('figure2_gaussian_uniform.png', format='png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':

    initial_grain_num = 20000
    # step_num = npy_file_aniso_000.shape[0]

    bin_width = 0.16  # Grain size distribution
    x_limit = [-0.5, 3.5]
    bin_num = round((abs(x_limit[0]) + abs(x_limit[1])) / bin_width)
    size_coordination = np.linspace((x_limit[0] + bin_width / 2), (x_limit[1] - bin_width / 2), bin_num)

    ################################################################################################################################## figure 3
    npy_file_aniso_mcp = np.load(r"T:\MF_new\prepare_paper_figure\figure3plot\mcp_plot.npy")
    npy_file_aniso_gaussian = np.load(r"T:\MF_new\Plot_gaussian_array.npy")
    npy_file_aniso_gaussian_square = np.load(r"T:\MF_new\Plot_gaussian_square_array.npy")
    npy_file_aniso_uniform = np.load(r"T:\MF_new\Plot_gaussian_uniform_array.npy")


    ### grain number [20000,12590,8369,447,300], change last one from 224 to 300
    A=[0,1,2,3,4]  ### MCP
    B=[0,1,2,3,4]  ### gaussian
    C=[0, 1, 2, 3, 4]         ### gaussian suqare
    D=[0, 1, 2, 3, 4]          ### uniform


    aniso_mag_A = []
    aniso_mag_B = []
    aniso_mag_C = []
    aniso_mag_D = []

    for a, b,c,d in zip(A, B,C,D):
        special_step_distribution_000 = a  # 2670/30 - 10 grains
        special_step_distribution_020 = b  # 2250/30 - 10 grains
        special_step_distribution_040 = c # 3480/30 - 10 grains
        special_step_distribution_060 = d # 3180/30 - 10 grains


        # Start polar figure
        plt.close()
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection='polar')

        ax.set_thetagrids(np.arange(0.0, 360.0, 45.0), fontsize=16)
        ax.set_thetamin(0.0)
        ax.set_thetamax(360.0)

        ax.set_rgrids(np.arange(0, 0.005, 0.0025))
        ax.set_rlabel_position(0.0)  # 标签显示在0°
        ax.set_rlim(0.0, 0.005)  # 标签范围为[0, 5000)
        ax.set_yticklabels(['0', '4e-3'], fontsize=16)

        ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow('True')

        # Aniso - 000
        newplace = np.transpose(npy_file_aniso_mcp[special_step_distribution_000, :, :, :], (1, 2, 0))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        slope_list = get_normal_vector_slope(P, sites, special_step_distribution_000, r"$MCP$")
        # aniso_mag, aniso_mag_stand = simple_magnitude_first_dimen(slope_list[:9])

        aniso_mag= simple_magnitude_abs(slope_list)
        aniso_mag_A.append(aniso_mag)

        newplace = np.transpose(npy_file_aniso_gaussian[special_step_distribution_020, :, :, :], (1, 2, 0))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        slope_list = get_normal_vector_slope(P, sites, special_step_distribution_020, r"$Gaussian$")
        # aniso_mag, aniso_mag_stand = simple_magnitude_first_dimen(slope_list[:9])

        aniso_mag= simple_magnitude_abs(slope_list)
        aniso_mag_B.append(aniso_mag)

        newplace = np.transpose(npy_file_aniso_gaussian_square[special_step_distribution_040, :, :, :], (1, 2, 0))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        slope_list = get_normal_vector_slope(P, sites, special_step_distribution_040, "Reshaped Gaussian")
        # aniso_mag, aniso_mag_stand = simple_magnitude_first_dimen(slope_list[:9])

        aniso_mag= simple_magnitude_abs(slope_list)
        aniso_mag_C.append(aniso_mag)

        newplace = np.transpose(npy_file_aniso_uniform[special_step_distribution_060, :, :, :], (1, 2, 0))
        P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
        slope_list = get_normal_vector_slope(P, sites, special_step_distribution_060, r"$Uniform$")
        # aniso_mag, aniso_mag_stand = simple_magnitude_first_dimen(slope_list[:9])

        aniso_mag= simple_magnitude_abs(slope_list)
        aniso_mag_D.append(aniso_mag)

        rounded_numbersA = [round(num, 3) for num in aniso_mag_A]
        rounded_numbersB = [round(num, 3) for num in aniso_mag_B]
        rounded_numbersC = [round(num, 3) for num in aniso_mag_C]
        rounded_numbersD = [round(num, 3) for num in aniso_mag_D]

        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=True) ### new
        # plt.savefig(current_path + "/figures/paper_figure3/normal_distribution_poly" + str(c) + ".png", dpi=400, bbox_inches='tight')
    #
    #
    #
    #
    #
    #
    # ################################################################################################################################## figure 4
    #
    # gaussian_uniform_0 = np.load(r"T:\MF_new\Plot_gaussian_square_array.npy")
    # gaussian_uniform_30 = np.load(r"T:\MF_new\Plot_gaussian_square30_array.npy")
    # gaussian_uniform_60 = np.load(r"T:\MF_new\Plot_gaussian_square60_array.npy")
    #
    # ### grain number [20000,12590,8369,447,300], change last one from 224 to 300
    # A = [0, 1, 2, 3, 4]  ### 30
    # B = [0, 1, 2, 3, 4]  ### 60
    # C = [0, 1, 2, 3, 4]  ### 90
    #
    # for a, b, c in zip(A, B, C):
    #     special_step_distribution_000 = a  # 2670/30 - 10 grains
    #     special_step_distribution_020 = b  # 2250/30 - 10 grains
    #     special_step_distribution_040 = c  # 3480/30 - 10 grains
    #
    #     # Start polar figure
    #     plt.close()
    #     fig = plt.figure(figsize=(5, 5))
    #     ax = fig.add_subplot(projection='polar')
    #
    #     ax.set_thetagrids(np.arange(0.0, 360.0, 45.0), fontsize=16)
    #     ax.set_thetamin(0.0)
    #     ax.set_thetamax(360.0)
    #
    #     ax.set_rgrids(np.arange(0, 0.005, 0.0025))
    #     ax.set_rlabel_position(0.0)  # 标签显示在0°
    #     ax.set_rlim(0.0, 0.005)  # 标签范围为[0, 5000)
    #     ax.set_yticklabels(['0', '4e-3'], fontsize=16)
    #
    #     ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    #     ax.set_axisbelow('True')
    #
    #     # Aniso - 000
    #     newplace = np.transpose(gaussian_uniform_0[special_step_distribution_000, :, :, :], (1, 2, 0))
    #     P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    #     slope_list = get_normal_vector_slope(P, sites, special_step_distribution_000, r"$0$")
    #
    #     newplace = np.transpose(gaussian_uniform_30[special_step_distribution_020, :, :, :], (1, 2, 0))
    #     P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    #     slope_list = get_normal_vector_slope(P, sites, special_step_distribution_020, r"$30$")
    #
    #     newplace = np.transpose(gaussian_uniform_60[special_step_distribution_040, :, :, :], (1, 2, 0))
    #     P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    #     slope_list = get_normal_vector_slope(P, sites, special_step_distribution_040, r"$60$")
    #
    #     # For bias
    #     xLim = [0, 360]
    #     binValue = 10.01
    #     binNum = round((abs(xLim[0]) + abs(xLim[1])) / binValue)
    #     freqArray_circle = np.ones(binNum)
    #     freqArray_circle = freqArray_circle / sum(freqArray_circle * binValue)
    #     slope_list_bias = freqArray_circle - slope_list
    #
    #     plt.legend(loc=(-0.24, -0.3), fontsize=16, ncol=3)
    #     plt.savefig(current_path + "/figures/paper_figure4/normal_distribution_poly" + str(c) + ".png", dpi=400, bbox_inches='tight')

    ################################################################################################################################## figure 5
    # gaussian_uniform_0 = np.load(r"T:\MF_new\Plot_gaussian_uniform_array.npy")
    # gaussian_uniform_30 = np.load(r"T:\MF_new\Plot_gaussian_uniform30_array.npy")
    # gaussian_uniform_60 = np.load(r"T:\MF_new\Plot_gaussian_uniform60_array.npy")
    #
    # ### grain number [20000,12590,8369,447,300], change last one from 224 to 300
    # A = [0, 1, 2, 3, 4]  ### 30
    # B = [0, 1, 2, 3, 4]  ### 60
    # C = [0, 1, 2, 3, 4]  ### 90
    #
    # for a, b, c in zip(A, B, C):
    #     special_step_distribution_000 = a  # 2670/30 - 10 grains
    #     special_step_distribution_020 = b  # 2250/30 - 10 grains
    #     special_step_distribution_040 = c  # 3480/30 - 10 grains
    #
    #     # Start polar figure
    #     plt.close()
    #     fig = plt.figure(figsize=(5, 5))
    #     ax = fig.add_subplot(projection='polar')
    #
    #     ax.set_thetagrids(np.arange(0.0, 360.0, 45.0), fontsize=16)
    #     ax.set_thetamin(0.0)
    #     ax.set_thetamax(360.0)
    #
    #     ax.set_rgrids(np.arange(0, 0.005, 0.0025))
    #     ax.set_rlabel_position(0.0)  # 标签显示在0°
    #     ax.set_rlim(0.0, 0.005)  # 标签范围为[0, 5000)
    #     ax.set_yticklabels(['0', '4e-3'], fontsize=16)
    #
    #     ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    #     ax.set_axisbelow('True')
    #
    #     # Aniso - 000
    #     newplace = np.transpose(gaussian_uniform_0[special_step_distribution_000, :, :, :], (1, 2, 0))
    #     P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    #     slope_list = get_normal_vector_slope(P, sites, special_step_distribution_000, r"$0$")
    #
    #     newplace = np.transpose(gaussian_uniform_30[special_step_distribution_020, :, :, :], (1, 2, 0))
    #     P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    #     slope_list = get_normal_vector_slope(P, sites, special_step_distribution_020, r"$30$")
    #
    #     newplace = np.transpose(gaussian_uniform_60[special_step_distribution_040, :, :, :], (1, 2, 0))
    #     P, sites, sites_list = get_normal_vector(newplace, initial_grain_num)
    #     slope_list = get_normal_vector_slope(P, sites, special_step_distribution_040, r"$60$")
    #
    #     # For bias
    #     xLim = [0, 360]
    #     binValue = 10.01
    #     binNum = round((abs(xLim[0]) + abs(xLim[1])) / binValue)
    #     freqArray_circle = np.ones(binNum)
    #     freqArray_circle = freqArray_circle / sum(freqArray_circle * binValue)
    #     slope_list_bias = freqArray_circle - slope_list
    #
    #     plt.legend(loc=(-0.24, -0.3), fontsize=16, ncol=3)
    #     plt.savefig(current_path + "/figures/paper_figure5/normal_distribution_poly" + str(c) + ".png", dpi=400, bbox_inches='tight')