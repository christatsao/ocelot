import torch
from tqdm import tqdm

@torch.inference_mode()
def evaluate(model, dataloader, criterion, device, amp):
    #Set to evaluation mode
    model.eval()
    metric_sum = 0
    n_samples = len(dataloader.dataset)

    #No need to waste memory resources for gradients if we are not using backpropagation
    with torch.autocast(device.type if device.type == 'cuda' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=n_samples, desc='Validation round', unit='batch', leave=False):
            images, true_masks = batch[0], batch[1]
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last if amp==True else torch.preserve_format)
            true_masks = true_masks.to(device=device, dtype=torch.float32)
            
            #Predict from image
            pred_masks = model(images)

            #Squeeze our masks
            true_masks = true_masks
            pred_masks = (torch.nn.functional.sigmoid(pred_masks) > 0.5).float()

            score = dice_coef(pred_masks.squeeze(1), true_masks.squeeze(1).float())
            
            metric_sum += score
    
    model.train()
    
    avg_metric = metric_sum/n_samples
    return avg_metric

def dice_coef(y_true, y_pred, smooth=1): #Implementation from https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2 
  intersection = torch.sum(y_true * y_pred, axis=[0,1])
  union = torch.sum(y_true, axis=[0,1]) + torch.sum(y_pred, axis=[0,1])
  dice = torch.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice.mean()