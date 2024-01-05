from os.path import exists

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Callable

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import MaskedPiecewiseRationalQuadraticAutoregressiveTransform

from simulator import Simulator


def get_default_flow():
    """Function to get a normalising flow object. Uses a simple default architecture.

    Returns:
        Flow: A neural spline flow
    """
    # We have 2-d vectors on a plane, so shape = features = 2.
    base_dist = StandardNormal(shape=[2])
    transform = MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
        features = 2,
        hidden_features=20,
        num_bins=10,
        tails='linear',
        tail_bound=10.,
        num_blocks=10,
        use_batch_norm=True
    )
    return Flow(transform, base_dist)


def get_dataloaders(file='end_points.pt', 
                    batch_size: int = 64, 
                    val_ratio: float = 0.1, 
                    save_data: bool = True, 
                    simulator_args: Tuple[int,...] = None) -> Tuple[DataLoader, DataLoader]:
    """Utility function for obtaining training and validation dataloaders

    Args:
        file (str, optional): File to load data from, if available. Defaults to 'end_points.pt'.
        batch_size (int, optional): Defaults to 64.
        val_ratio (float, optional): Fraction of the data to use for validation.  Defaults to 0.1.
        save_data (bool, optional): If True then new data generated will be saved to disk. Defaults to True.
        simulator_args (Tuple[int,...], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    


    if exists(file):
        data = torch.load(file)
    else:
        if simulator_args is None:
            simulator = Simulator()
        else:
            simulator = Simulator(*simulator_args)

        print('No data found, simulating 5000 new data points.')
        data = torch.stack([simulator()[0] for _ in tqdm(range(5000))])

        if save_data:
            torch.save(data, file)

    cut_off = round(len(data)*val_ratio)
    train_data = data[cut_off:]
    val_data = data[:cut_off]
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size, shuffle=True)

    return train_loader, val_loader


class Trainer:
    """Class for training a normalising flow.

    Attributes:
        density_estimator - a trainable estimator with 
        train_losses - a per epoch record of the training loss
        val_losses - a per epoch record of the validation loss.
    """
    def __init__(self,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 density_estimator: Module,
                 early_stop_bound: int = 20,
                 max_epochs: int = 500,
                 optimizer: Optimizer = Adam,
                 base_learning_rate: float = 0.01,
                 use_lr_scheduler: bool = True,
                 ) -> None:
        """
        Inits Trainer.

        Args:
            train_loader (DataLoader): dataloader class for training data.
            val_loader (DataLoader): dataloader class for validation data.
            density_estimator (Module): normalising flow or equivalent.
            early_stop_bound (int, optional): Used in early stopping condition - the number of 
            rounds of no improvement after which to stop. Defaults to 20.
            max_epochs (int, optional): For if early stopping does not occur. Defaults to 500.
            optimizer (Optimizer, optional): Defaults to Adam.
            base_learning_rate (float, optional): Defaults to 5e-3.
            use_lr_scheduler (bool, optional): If True, uses plateau rate scheduler. Defaults to True.
        """
        
        self.density_estimator = density_estimator
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._early_stop_bound = early_stop_bound
        self._max_epochs = max_epochs
        self._trained_epochs = 0

        optimisation_parameters = density_estimator.parameters()
        self._optimizer = optimizer(optimisation_parameters, lr=base_learning_rate)
        if use_lr_scheduler:
            self._lr_scheduler=ReduceLROnPlateau(self._optimizer, patience=25)

        self.train_losses = []
        self.val_losses = []


    def _loss(self, params: Tensor):
        """Cross-entropy loss (equivalent to KL divergence)

        Args:
            params (Tensor): paramters sampled from the underlying distribution.

        Returns:
            Tensor: value of loss.
        """
        return -torch.mean(self.density_estimator.log_prob(params))

    def training_loop(self, dataloader: DataLoader):
        """Executes a training epoch.

        Args:
            dataloader (DataLoader): Expects a dataloader with unlabelled data. 

        Returns:
            Tensor: mean loss
        """
        self.density_estimator.train()
        total_loss = 0.
        for params in dataloader:
            loss = self._loss(params)
            total_loss+=loss.detach()
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
        mean_loss = total_loss/len(dataloader.dataset) 
        return mean_loss
    
    def validation_loop(self, dataloader: DataLoader):
        """Checks validation loss

        Args:
            dataloader (DataLoader): Expects a dataloader with unlabelled data. 
        """
        self.density_estimator.eval()
        loss = self._loss(dataloader.dataset)
        return loss
    
    def log_and_print(self, train_loss, val_loss, since_improvement):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        print(f'\r epoch = {self._trained_epochs}, '
              f'train loss = {train_loss:.3e}, '
              f'val loss = {val_loss:.3e}, '
              f'epochs since improvement = {since_improvement}   ', end='')


    def train(self, epochs = 200):
        """Trains the density estimator. If the total epochs trained 
        already exceeds max_epochs raises an exception.

        Args:
            epochs (int, optional): Epochs to train for. Defaults to 200.

        Returns:
            Module: Trained normalising flow
        """
        if self._trained_epochs >=self._max_epochs:
            raise Exception(f"Already trained density estimator for the \
                            maximum number of epochs ({self._max_epochs})")

        best_loss = torch.inf
        rounds_since_improvement = 0
        for i in range(epochs):
            self._trained_epochs += 1
            train_loss = self.training_loop(self._train_loader)
            val_loss = self.validation_loop(self._val_loader)
            if val_loss < best_loss:
                best_loss = val_loss
                rounds_since_improvement = 0
            else:
                rounds_since_improvement+=1

            self.log_and_print(train_loss, val_loss, rounds_since_improvement)
            if rounds_since_improvement == self._early_stop_bound:
                break

        # Return for the sake of convenience. Also accessible as attribute of Trainer object.
        return self.density_estimator
