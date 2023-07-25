import torch
from tqdm import tqdm
from monai.losses import DiceCELoss, DiceLoss
from monai.networks.utils import one_hot
from comet_ml import Experiment


@torch.inference_mode()
def test(args, model, dataloader, device, amp, experiment=None):
    
    #Set model to evaluation mode
    model.eval()
    metric_sum = 0
    n_samples = len(dataloader.dataset)
    n_batches = n_samples / dataloader.batch_size

    if model.n_classes == 1:
        loss_metric = DiceCELoss(sigmoid=True)
    else:
        loss_metric = DiceCELoss(softmax=True, to_onehot_y=True)

    with torch.autocast(device.type if device.type == 'cuda' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=n_samples, desc='Validation round', unit='batch', leave=False):
            images, true_masks = batch[0], batch[1]
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last if amp==True else torch.preserve_format)
            true_masks = true_masks.to(device=device, dtype=torch.float32).unsqueeze(1)
            model.to(device)
            
            pred_masks = model(images).float()

            score = loss_metric(pred_masks, true_masks.float()).detach().cpu()

            metric_sum += torch.sum(score).item()
    model.train()
    
    #We divide by model.n_channels based on whether we are performing multiclass or binary
    avg_metric = (metric_sum/n_batches)/model.n_classes

    if experiment is not None:
        experiment.log_metric(avg_metric)
    return avg_metric
