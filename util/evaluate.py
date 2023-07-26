import torch
from tqdm import tqdm
from monai.losses import DiceLoss
from comet_ml import Experiment
from torchmetrics.classification import JaccardIndex, BinaryJaccardIndex
from monai.networks.utils import one_hot


@torch.inference_mode()
def evaluate(args, model, dataloader, device, epoch, experiment=None):
    
    #Set model to evaluation mode
    model.eval()
    metric_sum = 0
    n_samples = len(dataloader.dataset)
    n_batches = n_samples / dataloader.batch_size

    if model.n_classes == 1:
        metric = BinaryJaccardIndex().to(device)
    else:
        metric = JaccardIndex(task="multiclass", num_classes=model.n_classes).to(device)

    with torch.autocast(device.type if device.type == 'cuda' else 'cpu', enabled=args.amp):
        
        for batch in tqdm(dataloader, total=n_samples, desc='Validation round', unit='batch', leave=False):
            
            images, true_masks = batch[0], batch[1]
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last if args.amp==True else torch.preserve_format)
            true_masks = true_masks.to(device=device, dtype=torch.float32)
            model.to(device)
            pred_masks = model(images).float()
            
            if model.n_classes == 1:
                true_masks = true_masks.unsqueeze(1)
                pred_masks = torch.sigmoid(pred_masks)
            
            else:
                true_masks = one_hot(true_masks.unsqueeze(1), num_classes=model.n_classes)
                pred_masks = torch.softmax(pred_masks, dim=1)

            IOU = metric(pred_masks, true_masks.float()).detach().cpu()

            metric_sum += torch.sum(IOU).item()
    
    model.train()
    
    #We divide by model.n_channels based on whether we are performing multiclass or binary
    mIOU = (metric_sum/n_batches)

    if experiment is not None:
        experiment.log_metric('mIOU', mIOU, step=epoch)

    return mIOU
