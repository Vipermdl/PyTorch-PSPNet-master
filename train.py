import torch
from torch.autograd import Variable


class Train():
    """Performs the training of ``model`` given a training dataset data
    loader, the optimizer, and the loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to train.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - optim (``Optimizer``): The optimization algorithm.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - use_cuda (``bool``): If ``True``, the training is performed using
    CUDA operations (GPU).

    """

    def __init__(self, model, data_loader, optim, criterion, metric, device):
        self.data_loader = data_loader
        self.optim = optim
        self.metric = metric
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = [criterion_.to(self.device) for criterion_ in criterion]
        


    def _to_tensor(self, *tensors):
        t = []
        for __tensors in tensors:
            t.append(__tensors.to(self.device))
        return t

    def run_epoch(self, iteration_loss=True):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        self.model.train()
        epoch_loss = 0.0
        self.metric.reset()
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs, labels, class_labels = batch_data
            
            inputs, labels, class_labels = self._to_tensor(inputs, labels, class_labels)

            # Forward propagation
            outputs, classifiers = self.model(inputs)
            
            # Loss computation
            #loss = 0.6 * self.criterion[0](outputs, labels) + 0.4 * self.criterion[1](classifiers, class_labels)
            loss = self.criterion[0](outputs, labels)
            
            
            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            # Keep track of the evaluation metric
            self.metric.add(outputs.data, labels.data)

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader), self.metric.value()
