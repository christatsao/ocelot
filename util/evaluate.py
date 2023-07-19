import torch
from tqdm import tqdm
from monai.losses import DiceCELoss

@torch.inference_mode()
def evaluate(model, dataloader, device, amp):
    #Set to evaluation mode
    model.eval()
    metric_sum = 0
    n_samples = len(dataloader.dataset)
    loss_metric = DiceCELoss(sigmoid=True)

    #No need to waste memory resources for gradients if we are not using backpropagation
    with torch.autocast(device.type if device.type == 'cuda' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=n_samples, desc='Validation round', unit='batch', leave=False):
            images, true_masks = batch[0], batch[1]
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last if amp==True else torch.preserve_format)
            true_masks = true_masks.to(device=device, dtype=torch.float32)
            model.to(device)
            
            #Predict from imagenotebooks/JL_notebook_1.ipynb
            pred_masks = model(images)

            #Squeeze our masks
            pred_masks = pred_masks.float()

            score = loss_metric(pred_masks, true_masks.float()).detach().cpu()
            
            metric_sum += score.item()
    
    model.train()
    
    avg_metric = metric_sum/n_samples
    return avg_metric