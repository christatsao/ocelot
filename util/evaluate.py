import torch
from tqdm import tqdm
from torchmetrics.classification import JaccardIndex, ConfusionMatrix, Dice
from monai.networks.utils import one_hot

@torch.inference_mode()
def evaluate(args, model, dataloader, device):
    
    #Set model to evaluation mode
    model.eval()
    model.to(device)

    #Some variables that are only used by tqdm atm
    n_samples = len(dataloader.dataset)
    n_batches = n_samples / dataloader.batch_size
    
    #Lets try out mIoU and Confusion matrix as two metrics
    metric1 = JaccardIndex(task='multiclass', num_classes=model.outputChannel+1).to(device)
    metric2 = ConfusionMatrix(task='multiclass', num_classes=model.outputChannel+1).to(device)
    
    with torch.autocast(device.type if device.type == 'cuda' else 'cpu', enabled=args.amp):
        for batch in tqdm(dataloader, total=n_batches, desc='Evaluation round', unit='batch', leave=False):

            #Set up our model, images, and mask
            images, true_masks = batch[0], batch[1]
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last if args.amp==True else torch.preserve_format)
            true_masks = true_masks.to(device=device, dtype=torch.float32).unsqueeze(1)

            #pass our image to our model to generate prediction
            pred_masks = model(images)
            masks = pred_masks, true_masks
            
            #some formatting of our predictions as probabilities and ground truths to proper formatting with one_hot
            if model.outputChannel == 1:
                pred_masks, true_masks = binary_format(masks)
            
            else:
                pred_masks, true_masks = multiclass_format(masks, num_classes=model.outputChannel+1)

            metric1.update(pred_masks, true_masks)
            metric2.update(pred_masks, true_masks)

    model.train()
    
    metric1 = metric1.compute().detach().cpu()
    metric2 = normalize_cm(metric2.compute().detach().cpu())

    return metric1, metric2

def binary_format(masks):
    y_pred, y_true = masks

    #Apply activations and convert to one_hot format
    y_pred = torch.sigmoid(y_pred)
    y_pred = torch.where(y_pred > 0.5, 1.0, 0.0)
    y_pred = one_hot(y_pred, num_classes=2)

    #Convert our 1 dimensional GT mask to one_hot format
    y_true = one_hot(y_true, num_classes=2)

    return y_pred, y_true

def multiclass_format(masks, num_classes: int = 3):
    y_pred, y_true = masks

    #Apply activations and thresholding to predicted
    y_pred = torch.softmax(y_pred, dim=0)
    y_pred = torch.where(y_pred > 0.5, 1.0, 0.0)

    #Convert our 1 dimensional GT mask to one_hot format
    y_true = one_hot(y_true, num_classes=num_classes)

    return y_pred, y_true

def normalize_cm(convolution_matrix):
    #get sum of each row to divide later
    row_sum = torch.sum(convolution_matrix, dim=1, keepdim=True)

    #Avoid zero division error
    row_sum[row_sum == 0] = 1

    norm_convolution_matrix = convolution_matrix / row_sum
    
    return norm_convolution_matrix