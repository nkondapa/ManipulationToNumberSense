from training.runner_action_sizes import Runner
from model.networks import PreTrainedStackedEmbeddingNet, SiameseActionClassificationNet
import torch
import torch.utils.data
from torch import optim
import os

root = '../data/'
experiment_test_dataset_name = ''
experiment_output_path = '../trained_models/'
os.makedirs(experiment_output_path, exist_ok=True)


def run(experiment_name, seed, num_objects, action_size, sj, cj, noise_opt):
    print(experiment_name, seed)
    model, optimizer, scheduler, loss = generate_model(experiment_name)
    r = Runner(experiment_name)
    dataloader_params = {'num_batches': 30, 'batch_size': 180, 'experiment_name': 'temp',
                         'diameter_ranges': [(10, 30)],
                         'color_ranges': [(25, 256)],
                         'shape_types_options': [['square']],
                         'size_jitter': sj,
                         'color_jitter': cj,
                         'add_noise': noise_opt,
                         'num_objects': num_objects,
                         'seed': seed,
                         'action_size': action_size
                         }

    torch.manual_seed(seed)
    r.train_embedding_model(num_epochs=15, embedding_model=model, optimizer=optimizer,
                            scheduler=scheduler,
                            loss_fn=loss,
                            dataloader_params=dataloader_params,
                            model_save_directory=experiment_output_path)


def generate_model(experiment_name):
    lr = 0.0001
    embedding_net1 = PreTrainedStackedEmbeddingNet(2)
    model1 = SiameseActionClassificationNet(embedding_net1, unique_id=experiment_name)
    loss_fn1 = torch.nn.NLLLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=lr)
    scheduler = None

    return model1, optimizer1, scheduler, loss_fn1


seeds = range(8)
for cj, sj in [(5, 3)]:
    for num_obj, act_size in [(3, 3), (5, 3), (5, 5)]:
        for seed in seeds:
            exp_name = f'model2_{seed}_num_obj={num_obj}_action_size={act_size}'
            run(exp_name, seed, num_objects=num_obj, action_size=act_size, sj=sj, cj=cj, noise_opt=False)
