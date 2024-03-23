import torch
import torchvision.transforms as transforms
import os
import json
import pickle as pkl

# this data loader generates a dataset according to the parameters passed, it doesn't require
# compression or storage of many images (the exact training data will not persist).
from data_loaders.online_sequential_pair_data_loader_action_sizes import OnlineSequentialPairDataLoader


class Runner:

    def __init__(self, experiment_name=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_history = []

        self.experiment_name = experiment_name

    @staticmethod
    def generate_model_save_directory(current_save_dir, model):

        current_save_dir += model.name.replace(' ', '_') + '/' + \
                            model.unique_id + '/' + 'trained_by=' + 'online_sequential_data_loader' + '/'
        return current_save_dir

    def train_embedding_model_epoch(self, embedding_model, optimizer, loss_fn, lr_scheduler, epoch,
                                    dataloader_params):
        embedding_model.train()

        trans = transforms.Compose([transforms.ToTensor()])
        batch_size = dataloader_params['batch_size']
        dataloader_params['experiment_name'] = self.experiment_name
        dataloader_params['epoch'] = epoch

        # load a novel sequence of events from the "motor" system
        loader_obj = OnlineSequentialPairDataLoader(**dataloader_params, transforms=trans)
        train_data_loader = torch.utils.data.DataLoader(loader_obj, batch_size=batch_size)

        for batch_idx, (data, targets) in enumerate(train_data_loader):
            for k in range(len(data)):
                data[k] = data[k].to(self.device)
            targets = targets.type(torch.LongTensor).to(self.device)
            optimizer.zero_grad()

            target = targets
            model = embedding_model
            output = model(data[0], data[1])

            target = torch.squeeze(target[:, 0])

            loss_same = loss_fn(output, torch.argmax(target, dim=1))

            loss = loss_same
            loss.backward()

            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * train_data_loader.batch_size, len(train_data_loader.dataset),
                           100. * batch_idx * train_data_loader.batch_size / len(train_data_loader.dataset),
                    loss.item()))
                self.loss_history.append(loss.cpu().detach().numpy())

    def train_embedding_model(self, embedding_model, num_epochs=10,
                              optimizer=None, loss_fn=None, scheduler=None, dataloader_params=None,
                              model_save_directory=None,
                              save_frequency=4):

        assert dataloader_params is not None

        # send model to GPU if available
        embedding_model.to(self.device)

        # generate save directory
        model_save_directory = self.generate_model_save_directory(model_save_directory, embedding_model)
        os.makedirs(model_save_directory, exist_ok=True)

        # save experiment name : model save directory
        exp_dict_path = '../trained_models' + '/experiment_dict.pkl'
        if os.path.exists(exp_dict_path):
            exp_dict = pkl.load(open(exp_dict_path, 'rb'))
        else:
            exp_dict = {}

        exp_dict[self.experiment_name] = model_save_directory
        with open(exp_dict_path, 'wb') as f:
            pkl.dump(exp_dict, f)

        with open(model_save_directory + 'params.txt', 'w') as f:
            json.dump(dataloader_params, f)

        # final model save path
        final_path = model_save_directory + '/final.pt'

        base_seed = None
        if dataloader_params['seed'] is not None:
            base_seed = dataloader_params['seed'] * num_epochs

        for epoch in range(num_epochs):
            if dataloader_params['seed'] is not None:
                dataloader_params['seed'] = base_seed + epoch

            self.train_embedding_model_epoch(embedding_model=embedding_model,
                                             optimizer=optimizer,
                                             loss_fn=loss_fn,
                                             lr_scheduler=scheduler,
                                             epoch=epoch,
                                             dataloader_params=dataloader_params)

            # if there is a scheduler, start it
            if scheduler is not None:
                scheduler.step()

            # snapshots of model
            if epoch % save_frequency == 0 and epoch != 0:
                checkpoint = {
                    'epoch': epoch,
                    'model': embedding_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': scheduler,
                    'loss_history': self.loss_history}
                torch.save(checkpoint, model_save_directory + '/{:03}.pt'.format(epoch))

        final = {
            'epoch': 'final',
            'model': embedding_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_sched': scheduler}

        torch.save(final, final_path)

        return embedding_model, model_save_directory
