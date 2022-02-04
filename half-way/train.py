import neptune.new as neptune

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model_bn import SSD300, MultiBoxLoss
from datasets import Dataset
from utils import *

data_folder_Thermal = './data_list/Thermal/'
data_folder_RGB='./data_list/RGB/'
keep_difficult = False 

n_classes = len(label_map) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint =None
batch_size = 16 
iterations = 120000 
workers = 4 
print_freq = 50  
lr = 5e-4 
decay_lr_at = [80000, 100000] 
decay_lr_to = 0.1  
momentum = 0.9 
weight_decay = 5e-4 
grad_clip = None  
cudnn.benchmark = True

def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)

        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.Adam(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    train_dataset = Dataset(data_folder_RGB,data_folder_Thermal,
                                     split='train',
                                     keep_difficult=keep_difficult)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True) 


    epochs = iterations // (len(train_dataset) // 16)
    decay_lr_at = [it // (len(train_dataset) // 16) for it in decay_lr_at]

    parameters={
      'learning_rate':lr,
      'batch_size':batch_size,
      'n_epochs':epochs,
      'momentum':momentum,
      'weight_decay':weight_decay
    }
    run['model/parameters']=parameters

    for epoch in range(start_epoch, epochs):

        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)
        run['model/saved_model'].upload('final_checkpoint.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()  

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter() 

    start = time.time()

    for i, (images_rgb,images_thermal, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        images_rgb = images_rgb.to(device) 
        images_thermal = images_thermal.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images_rgb,images_thermal) 

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels) 

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()


        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images_rgb.size(0))
        batch_time.update(time.time() - start)

        start = time.time()


        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))


        run['train/epoch/loss'].log(losses.val)
        run['train/epoch/avg_loss'].log(losses.avg)

    del predicted_locs, predicted_scores, images_rgb,images_thermal, boxes, labels 


if __name__ == '__main__':
    main()
