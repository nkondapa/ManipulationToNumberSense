from plotting.dataset_plots import *
import os

FIGURE_OUTPUT_FOLDER = '../figures/'
FOLDER_PATH = FIGURE_OUTPUT_FOLDER + '/dataset_figures/'
os.makedirs(FOLDER_PATH, exist_ok=True)

# # Sequence Plots
# dataloader_params1 = {'num_batches': 1, 'batch_size': 180, 'experiment_name': 'temp',
#                       'diameter_ranges': [(15, 16)],
#                       'color_ranges': [(255, 256)],
#                       'shape_types_options': [['square']],
#                       'num_objects': 3,
#                       'seed': 10,
#                       }
#
# dataloader_params2 = {'num_batches': 1, 'batch_size': 180, 'experiment_name': 'temp',
#                       'diameter_ranges': [(10, 30)],
#                       'color_ranges': [(25, 255)],
#                       'shape_types_options': [['square']],
#                       'num_objects': 3,
#                       'seed': 10,
#                       }
#
# dataloader_params3 = {'num_batches': 30, 'batch_size': 180, 'experiment_name': 'temp',
#                       'diameter_ranges': [(10, 30)],
#                       'color_ranges': [(25, 256)],
#                       'shape_types_options': [['square']],
#                       'size_jitter': 3,
#                       'color_jitter': 5,
#                       'add_noise': False,
#                       'num_objects': 3,
#                       'seed': 10,
#                       }
#
# dataloader_params4 = {'num_batches': 30, 'batch_size': 180, 'experiment_name': 'temp',
#                       'diameter_ranges': [(10, 30)],
#                       'color_ranges': [(25, 256)],
#                       'shape_types_options': [['square']],
#                       'size_jitter': None,
#                       'color_jitter': None,
#                       'add_noise': False,
#                       'num_objects': 3,
#                       'seed': 10,
#                       'action_size': 3
#                       }
#
# fig1 = generate_dataset_sample(dataloader_params1)
# fig2 = generate_dataset_sample(dataloader_params2)
# fig3 = generate_dataset_sample(dataloader_params3)
# fig4 = generate_dataset_sample(dataloader_params4)
#
# ax1 = fig1.gca()
# ax1.annotate(s='A', fontsize=17, weight='bold', color='white', xy=(0.016, 0.83), xycoords='figure fraction')
# fig1.savefig(FOLDER_PATH + 'dataset1_sequence.pdf')
#
# ax2 = fig2.gca()
# ax2.annotate(s='B', fontsize=17, weight='bold', color='white', xy=(0.016, 0.83), xycoords='figure fraction')
# fig2.savefig(FOLDER_PATH + 'dataset2_sequence.pdf')
#
# ax3 = fig3.gca()
# # ax3.annotate(s='C', fontsize=17, weight='bold', color='white', xy=(0.016, 0.83), xycoords='figure fraction')
# fig3.savefig(FOLDER_PATH + 'dataset3_sequence.pdf')
#
# ax4 = fig4.gca()
# # ax3.annotate(s='C', fontsize=17, weight='bold', color='white', xy=(0.016, 0.83), xycoords='figure fraction')
# fig3.savefig(FOLDER_PATH + 'dataset4_sequence.pdf')
#
# fig1 = generate_dataset_sample(dataloader_params1, invert_colors=True)
# fig2 = generate_dataset_sample(dataloader_params2, invert_colors=True)
# fig3 = generate_dataset_sample(dataloader_params3, invert_colors=True)
# fig4 = generate_dataset_sample(dataloader_params4, invert_colors=True)
#
# ax1 = fig1.gca()
# ax1.annotate(s='A', fontsize=17, weight='bold', color='black', xy=(0.016, 0.83), xycoords='figure fraction')
# fig1.savefig(FOLDER_PATH + 'dataset1_sequence_inverted.pdf')
#
# ax2 = fig2.gca()
# ax2.annotate(s='B', fontsize=17, weight='bold', color='black', xy=(0.016, 0.83), xycoords='figure fraction')
# fig2.savefig(FOLDER_PATH + 'dataset2_sequence_inverted.pdf')
#
# ax3 = fig3.gca()
# # ax3.annotate(s='C', fontsize=17, weight='bold', color='white', xy=(0.016, 0.83), xycoords='figure fraction')
# fig3.savefig(FOLDER_PATH + 'dataset3_sequence_inverted.pdf')
#
# ax4 = fig4.gca()
# # ax3.annotate(s='C', fontsize=17, weight='bold', color='white', xy=(0.016, 0.83), xycoords='figure fraction')
# fig4.savefig(FOLDER_PATH + 'dataset4_sequence_inverted.pdf')

# Statistics Figures
fig1, ax1 = plt.subplots(1, 1, constrained_layout=True)
fig2, ax2 = plt.subplots(1, 1, constrained_layout=True)
fig3, ax3 = plt.subplots(1, 1, constrained_layout=True)
fig4, ax4 = plt.subplots(1, 1, constrained_layout=True)

generate_train_set_statistics('model1_0', ax1)
generate_train_set_statistics('model2_0', ax2)
generate_train_set_statistics(f'model2_{0}_noise={False}_cj={5}_sj={3}', ax3)
generate_train_set_statistics(f'model2_0_num_obj=3_action_size=3', ax4)

ax1.set_ylabel('Mean image intensity', fontsize=16)
ax1.set_xlabel('Number of objects', fontsize=16)
ax2.set_xlabel('Number of objects', fontsize=16)
ax3.set_xlabel('Number of objects', fontsize=16)
ax3.set_ylabel('Mean image intensity', fontsize=16)
ax4.set_xlabel('Number of objects', fontsize=16)

ax1 = fig1.gca()
ax1.annotate(s='A', fontsize=25, weight='bold', xy=(0.15, 0.925), xycoords='figure fraction')
plt.setp(ax1.get_xticklabels(), fontsize=14)
plt.setp(ax1.get_yticklabels(), fontsize=14)
ax1.set_ylim([0, 0.040])
fig1.savefig(FOLDER_PATH + 'dataset1_statistics.pdf')

ax2 = fig2.gca()
ax2.annotate(s='B', fontsize=25, weight='bold', xy=(0.1, 0.925), xycoords='figure fraction')
plt.setp(ax2.get_xticklabels(), fontsize=14)
plt.setp(ax2.get_yticklabels(), fontsize=14)
ax2.set_ylim([0, 0.040])
fig2.savefig(FOLDER_PATH + 'dataset2_statistics.pdf')

ax3 = fig3.gca()
ax3.annotate(s='C', fontsize=25, weight='bold', xy=(0.15, 0.925), xycoords='figure fraction')
plt.setp(ax3.get_xticklabels(), fontsize=14)
plt.setp(ax3.get_yticklabels(), fontsize=14)
ax3.set_ylim([0, 0.040])
fig3.savefig(FOLDER_PATH + 'dataset3_statistics.pdf')

ax4 = fig4.gca()
ax4.annotate(s='D', fontsize=25, weight='bold', xy=(0.1, 0.925), xycoords='figure fraction')
plt.setp(ax4.get_xticklabels(), fontsize=14)
plt.setp(ax4.get_yticklabels(), fontsize=14)
ax4.set_ylim([0, 0.040])
fig4.savefig(FOLDER_PATH + 'dataset4_statistics.pdf')