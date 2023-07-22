import torch
from tqdm import tqdm
from monai.losses import DiceCELoss, DiceLoss

@torch.inference_mode()
def evaluate(args, model, dataloader, device, amp):
    #Set to evaluation mode
    model.eval()
    metric_sum = 0
    n_samples = len(dataloader.dataset)
    n_batches = n_samples / dataloader.batch_size
    loss_metric = DiceCELoss(sigmoid=True) if model.n_channels == 1 else DiceLoss(to_onehot_y=True)

    #No need to waste memory resources for gradients if we are not using backpropagation
    with torch.autocast(device.type if device.type == 'cuda' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=n_samples, desc='Validation round', unit='batch', leave=False):
            images, true_masks = batch[0], batch[1]
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last if amp==True else torch.preserve_format)
            true_masks = true_masks.to(device=device, dtype=torch.float32).unsqueeze(1)
            model.to(device)
            
            pred_masks = model(images)

            #Squeeze our masks
            pred_masks = pred_masks.float()

            score = loss_metric(pred_masks, true_masks.float()).detach().cpu()
            
            metric_sum += score.item()
    
    model.train()
    
    avg_metric = metric_sum/n_batches
    return avg_metric