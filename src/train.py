import torch
from torch import nn
from torch.utils.data import Dataloader


def set_train(*modules: nn.Module) -> None:
    for module in modules:
        module.train()


def set_eval(*modules: nn.Module) -> None:
    for module in modules:
        module.eval()


def step(model: nn.module, classifier: nn.module, 
         batch: tuple[torch.Tensor], crossmodal_criterion: nn.module, 
         classification_criterion: torch.Module, alpha: float) -> torch.Tensor:

    image, signal, y = batch

    out = model(image, signal)
    loss = alpha * crossmodal_criterion(*out)

    x = torch.cat(out, 0)
    y = torch.tile(y, (2,))
    out = classifier(x)
    loss += (1 - alpha) * classification_criterion(out, y)
    
    return loss


@torch.no_grad()
def validate(model: nn.Module, classifier: nn.Module, 
             crossmodal_criterion: nn.Module, 
             classification_criterion: nn.Module, 
             dataloader: Dataloader, alpha: float) -> torch.Tensor:

    set_eval(model, classifier, crossmodal_criterion)
    loss = 0
    for batch in dataloader:
        loss += step(model, classifier, batch, crossmodal_criterion,
                     classification_criterion, alpha).detach().item()

    return loss / len(dataloader)


def train(model: nn.Module, classifier: nn.Module, 
          crossmodal_criterion: nn.Module, 
          classification_criterion: nn.Module, 
          train_loader: Dataloader, alpha: float):
    pass
