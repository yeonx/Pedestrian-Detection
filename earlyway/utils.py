import json
import os
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

voc_labels = ['person']
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}

distinct_colors = ['#3cb44b', '#0082c8']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}

def decimate(tensor, m):
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):

    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties) 
    n_classes = len(label_map)

    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):

        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)

        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]

            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    mean_average_precision = average_precisions.mean().item()

    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision


def xy_to_cxcy(xy):
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def find_intersection(set_1, set_2):
    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)
    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)
    return intersection / union  # (n1, n2)

def expand(image_rgb,image_thermal, boxes, filler_rgb,filler_thermal):
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h_rgb = image_rgb.size(1)
    original_w_rgb = image_rgb.size(2)
    original_h_thermal = image_thermal.size(1)
    original_w_thermal = image_thermal.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h_rgb = int(scale * original_h_rgb)
    new_w_rgb = int(scale * original_w_rgb)
    new_h_thermal = int(scale * original_h_thermal)
    new_w_thermal = int(scale * original_w_thermal)

    filler_rgb = torch.FloatTensor(filler_rgb)
    filler_thermal = torch.FloatTensor(filler_thermal)
    new_image_rgb = torch.ones((3, new_h_rgb, new_w_rgb), dtype=torch.float) * filler_rgb.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)#######################
    new_image_thermal = torch.ones((1, new_h_thermal, new_w_thermal), dtype=torch.float) * filler_thermal.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)#######################
    
    left_rgb = random.randint(0, new_w_rgb - original_w_rgb)
    right_rgb = left_rgb + original_w_rgb
    top_rgb = random.randint(0, new_h_rgb - original_h_rgb)
    bottom_rgb = top_rgb + original_h_rgb
    new_image_rgb[:, top_rgb:bottom_rgb, left_rgb:right_rgb] = image_rgb

    left_thermal = random.randint(0, new_w_thermal - original_w_thermal)
    right_thermal = left_thermal + original_w_thermal
    top_thermal = random.randint(0, new_h_thermal - original_h_thermal)
    bottom_thermal = top_thermal + original_h_thermal
    new_image_thermal[:, top_thermal:bottom_thermal, left_thermal:right_thermal] = image_thermal

    # Adjust bounding boxes' coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([left_rgb, top_rgb, left_rgb, top_rgb]).unsqueeze(0)  # (n_objects, 4), n_objects is the no. of objects in this image
    return new_image_rgb,new_image_thermal, new_boxes


def random_crop(image_rgb,image_thermal, boxes, labels, difficulties):
    original_h = image_rgb.size(1)
    original_w = image_rgb.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping
        # If not cropping
        if min_overlap is None:
            return image_rgb,image_thermal, boxes, labels, difficulties

        max_trials = 50
        for _ in range(max_trials):

            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)
            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue
            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)
            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)
            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue
            # Crop image
            new_image_rgb = image_rgb[:, top:bottom, left:right]  # (3, new_h, new_w)
            new_image_thermal = image_thermal[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)
            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index
            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue
            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]
            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]
            return new_image_rgb,new_image_thermal, new_boxes, new_labels, new_difficulties


def flip(image_rgb,image_thermal, boxes):

    new_image_rgb = FT.hflip(image_rgb)
    new_image_thermal = FT.hflip(image_thermal)

    new_boxes = boxes
    new_boxes[:, 0] = image_rgb.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image_rgb.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]
    return new_image_rgb,new_image_thermal, new_boxes


def resize(image_rgb,image_thermal, boxes, dims=(300, 300), return_percent_coords=True):

    new_image_rgb = FT.resize(image_rgb, dims)
    new_image_thermal = FT.resize(image_thermal, dims)

    old_dims = torch.FloatTensor([image_rgb.width, image_rgb.height, image_rgb.width, image_rgb.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates
    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims
    return new_image_rgb,new_image_thermal, new_boxes


def photometric_distort(image):
    new_image = image
    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]
    random.shuffle(distortions)
    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)
            new_image = d(new_image, adjust_factor)
    return new_image

def transform(image_rgb,image_thermal, boxes, labels, difficulties, split):
    assert split in {'TRAIN', 'TEST'}
    new_image_rgb = image_rgb
    new_image_thermal = image_thermal
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties

    if split == 'TRAIN':
        mean_thermal = [0.449]
        std_thermal = [0.226]
        mean_rgb = [0.485, 0.456, 0.406]
        std_rgb = [0.229, 0.224, 0.225]

        new_image_rgb = photometric_distort(new_image_rgb)
        new_image_rgb = FT.to_tensor(new_image_rgb)

        new_image_thermal = photometric_distort(new_image_thermal)
        new_image_thermal = FT.to_tensor(new_image_thermal)

        if random.random() < 0.5:
            new_image_rgb,new_image_thermal, new_boxes = expand(new_image_rgb,new_image_thermal, boxes, filler_rgb=mean_rgb,filler_thermal=mean_thermal)
        new_image_rgb,new_image_thermal, new_boxes, new_labels, new_difficulties = random_crop(new_image_rgb,new_image_thermal, new_boxes, new_labels,
                                                                         new_difficulties)
        new_image_rgb= FT.to_pil_image(new_image_rgb)
        new_image_thermal = FT.to_pil_image(new_image_thermal)

  
        if random.random() < 0.5:
            new_image_rgb,new_image_thermal, new_boxes = flip(new_image_rgb,new_image_thermal, new_boxes)
    
    new_image_rgb,new_image_thermal, new_boxes = resize(new_image_rgb,new_image_thermal, new_boxes, dims=(300, 300))
    new_image_rgb = FT.to_tensor(new_image_rgb)
    new_image_thermal = FT.to_tensor(new_image_thermal)

    new_image_rgb = FT.normalize(new_image_rgb,mean=mean_rgb, std=std_rgb)
    new_image_thermal = FT.normalize(new_image_thermal, mean=mean_thermal,std=std_thermal)
    
    return new_image_rgb,new_image_thermal, new_boxes, new_labels, new_difficulties


def adjust_learning_rate(optimizer, scale):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))

def accuracy(scores, targets, k):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def save_checkpoint(epoch, model, optimizer):
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_earlyway.pth.tar'
    torch.save(state, filename)


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
