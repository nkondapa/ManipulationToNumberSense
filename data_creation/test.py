import numpy as np
import matplotlib.pyplot as plt
import random
import cvxpy as cp

# N = range(1, 31)
# target_total_intensity = 255 * 29 ** 2
# sizes = range(10, 30)
# total_intensity_list = []
# intensity_dict = {}
# image_params = []
# image_params_dict = {}
# for ni in N:
#     n_intensity_list = []
#     n_image_params = []
#     for s in sizes:
#         intensity = target_total_intensity / (ni * s**2)
#         # s = np.sqrt(target_total_intensity / (ni * intensity))
#         # print(ni, intensity, s, round(s), round(s)**2 * intensity * ni)
#         if 25 <= intensity <= 255:
#             total_intensity = int(round(s)**2) * intensity * ni
#             total_intensity_list.append(total_intensity)
#             n_intensity_list.append(total_intensity)
#             image_params.append((ni, round(s), intensity))
#             n_image_params.append((ni, round(s), intensity))
#     intensity_dict[ni] = n_intensity_list
#     image_params_dict[ni] = n_image_params
#
# print(image_params)
#
# for k in N:
#     inds = list(range(len(intensity_dict[k])))
#     print(len(inds), image_params[k])
#     # plt.scatter(np.zeros(len(sinds)) + k, intensity_dict[k][sinds])
#
# plt.show()

# N = range(4, 29)
# target_surface_area = 29 * 10**2
# surface_area = []
# sizes_list = []
# max_iterations = 1000
# for ni in N:
#     s = np.sqrt(target_surface_area / ni)
#     lb = 10 - s
#     ub = 29 - s
#
#     success = False
#     while not success:
#         diffv = np.zeros(ni)
#         sv = np.zeros(ni) + s
#         inds = list(range(ni))
#         print(ni)
#         # print(lb, ub)
#         for i in range(len(diffv)):
#             val = np.random.random() * (ub - lb) + lb
#             # print(val)
#             diffv[i] += val
#
#             # rebalance
#             rebalance_remainder = -val
#             count = 0
#             while abs(rebalance_remainder) > 0:
#                 count += 1
#                 # print(rebalance_remainder)
#                 if abs(rebalance_remainder) <= 0.05:
#                     num_inds = 1
#                 else:
#                     num_inds = np.random.randint(1, ni)
#
#                 sinds = np.random.choice(inds, num_inds)
#                 for sind in sinds:
#                     if lb < rebalance_remainder/len(sinds) + diffv[sind] < ub:
#                         diffv[sind] = rebalance_remainder / len(sinds) + diffv[sind]
#                         rebalance_remainder -= rebalance_remainder/len(sinds)
#         sv += diffv
#         if abs(np.sum(sv ** 2) - target_surface_area) <= 100:
#             success = True
#             print(sv)
#             surface_area.append(np.sum(sv ** 2))
    # print(np.sum(sv), np.sum(diffv), diffv)
    # print(sv, np.round(sv), np.sum(np.round(sv)))
    # print(ni, s, ni * round(s) ** 2, lb, ub)

# print(sizes)
#
#
# plt.scatter(N, surface_area)
# plt.show()

# N = range(1, 30)
# target_surface_area = 25*2 * 20
# for ni in N:
#     avg_size = int(np.sqrt(target_surface_area / ni))
#     if avg_size < 10 or avg_size > 29:
#         print(ni, 'skip')
#         continue
#
#     f = lambda x: avg_size ** 2 * x + (avg_size + 1) ** 2 * (ni - x)
#
#     best_x = -1
#     min_diff = target_surface_area
#     for x in range(0, ni+1):
#         new_min = abs(f(x) - target_surface_area)
#         if min_diff > new_min:
#             min_diff = new_min
#             best_x = x
#
#     size_vector = np.zeros(ni) + avg_size
#     inds = np.arange(0, ni)
#     sinds = np.random.choice(inds, best_x)
#     size_vector[sinds] += 1
#
#

surface_areas = np.arange(10, 30) ** 2
# x = cp.Variable(len(surface_areas), boolean=True)
target_surface_area = 20**2 * 20
N = range(1, 30)
inds = np.arange(len(surface_areas))
for ni in N:
    selection_vector = cp.Variable(20, integer=True)
    obj = cp.Minimize((selection_vector @ surface_areas - target_surface_area))
    constraints = [cp.sum(selection_vector) == ni]
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=True)
    print(selection_vector.value)
