from training.runner import Runner
from model.networks import PreTrainedStackedEmbeddingNet, SiameseActionClassificationNet
import torch
import torch.utils.data
from torch import optim
import os

root = '../data/'
experiment_test_dataset_name = ''
experiment_output_path = '../trained_models/'
os.makedirs(experiment_output_path, exist_ok=True)


def run(experiment_name, seed, emb_dim, sj, cj, noise_opt):
    print(experiment_name, seed)
    model, optimizer, scheduler, loss = generate_model(experiment_name, emb_dim)
    r = Runner(experiment_name)
    dataloader_params = {'num_batches': 30, 'batch_size': 180, 'experiment_name': 'temp',
                         'diameter_ranges': [(10, 30)],
                         'color_ranges': [(25, 256)],
                         'shape_types_options': [['square']],
                         'size_jitter': sj,
                         'color_jitter': cj,
                         'add_noise': noise_opt,
                         'num_objects': 3,
                         'seed': seed,
                         }

    r.train_embedding_model(num_epochs=30, embedding_model=model, optimizer=optimizer,
                            scheduler=scheduler,
                            loss_fn=loss,
                            dataloader_params=dataloader_params,
                            model_save_directory=experiment_output_path)


def generate_model(experiment_name, emb_dim):
    lr = 0.0001
    embedding_net1 = PreTrainedStackedEmbeddingNet(emb_dim)
    model1 = SiameseActionClassificationNet(embedding_net1, unique_id=experiment_name)
    loss_fn1 = torch.nn.NLLLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=lr)
    scheduler = None

    return model1, optimizer1, scheduler, loss_fn1


seed = 0
for cj, sj in [(5, 3)]:
    for ed in [1, 2, 8, 16, 64, 256]:
        exp_name = f'model2_ed={ed}_noise={False}_cj={cj}_sj={sj}'
        run(exp_name, seed, ed, sj=sj, cj=cj, noise_opt=False)
