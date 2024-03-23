from model.networks import PreTrainedStackedEmbeddingNet, SiameseActionClassificationNet
import torch
import pickle as pkl
import cv2
import numpy as np
from global_variables import ROOT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# search experiment dict for path to saved experiment by experiment name
def load_trained_model(experiment_name, model_name='final.pt'):
    with open(f'{ROOT}/trained_models/experiment_dict.pkl', 'rb') as f:
        experiment_dict = pkl.load(f)

    model_save_directory = experiment_dict[experiment_name].replace('..', ROOT)
    print(model_save_directory)
    embedding_dimension = int(model_save_directory.split('embedding_net_')[1].split('/')[0])
    embedding_net1 = PreTrainedStackedEmbeddingNet(embedding_dimension)
    model = SiameseActionClassificationNet(embedding_net1, 'dummy')
    model.load_state_dict(torch.load(model_save_directory + f'{model_name}')['model'])
    model.to(device)

    return model


def initialize_video_writer(filepath, fps=30, capsize=(640, 480), fourcc_code='mp4v'):

    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
    out = cv2.VideoWriter()
    success = out.open(filename=filepath, fourcc=fourcc, fps=fps, frameSize=capsize, isColor=True)

    return out, success


def write_pyplot_fig_to_video_writer(videowriter, fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)
    buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    videowriter.write(buf)


def write_figure_list_to_video_writer(filepath, figure_list, fps=30, capsize=(640, 480), fourcc_code='mp4v'):

    videowriter, success = initialize_video_writer(filepath, fps, capsize, fourcc_code)
    if not success:
        raise Exception('Failed to open videowriter')

    for fig in figure_list:
        write_pyplot_fig_to_video_writer(videowriter, fig)

    close_video_writer(videowriter)


def close_video_writer(out):
    out.release()