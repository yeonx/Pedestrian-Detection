from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGGBase(nn.Module):
    def __init__(self):
        super(VGGBase, self).__init__()
        """
        RGB
        """
        self.conv1_1_rgb = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_1_bn_rgb = nn.BatchNorm2d(64)
        self.conv1_2_rgb = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv1_2_bn_rgb = nn.BatchNorm2d(64)
        self.pool1_rgb = nn.MaxPool2d(kernel_size=2, stride=2) #300->150

        self.conv2_1_rgb = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_1_bn_rgb = nn.BatchNorm2d(128)
        self.conv2_2_rgb = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv2_2_bn_rgb = nn.BatchNorm2d(128)
        self.pool2_rgb = nn.MaxPool2d(kernel_size=2, stride=2) #150->75

        self.conv3_1_rgb = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_1_bn_rgb = nn.BatchNorm2d(256)
        self.conv3_2_rgb = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_2_bn_rgb = nn.BatchNorm2d(256)
        self.conv3_3_rgb = nn.Conv2d(256, 256, kernel_size=3, padding=1) #75->38
        self.conv3_3_bn_rgb = nn.BatchNorm2d(256)
        self.pool3_rgb = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1_rgb = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_1_bn_rgb = nn.BatchNorm2d(512)
        self.conv4_2_rgb = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_2_bn_rgb = nn.BatchNorm2d(512)
        self.conv4_3_rgb = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3_bn_rgb = nn.BatchNorm2d(512)

        """
        Thermal
        """
        self.conv1_1_thermal = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv1_1_bn_thermal = nn.BatchNorm2d(64)
        self.conv1_2_thermal = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv1_2_bn_thermal = nn.BatchNorm2d(64)
        self.pool1_thermal = nn.MaxPool2d(kernel_size=2, stride=2) #300->150

        self.conv2_1_thermal = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_1_bn_thermal = nn.BatchNorm2d(128)
        self.conv2_2_thermal = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv2_2_bn_thermal = nn.BatchNorm2d(128)
        self.pool2_thermal = nn.MaxPool2d(kernel_size=2, stride=2) #150->75

        self.conv3_1_thermal = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_1_bn_thermal = nn.BatchNorm2d(256)
        self.conv3_2_thermal = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_2_bn_thermal = nn.BatchNorm2d(256)
        self.conv3_3_thermal = nn.Conv2d(256, 256, kernel_size=3, padding=1) #75->38
        self.conv3_3_bn_thermal = nn.BatchNorm2d(256)
        self.pool3_thermal = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1_thermal = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_1_bn_thermal = nn.BatchNorm2d(512)
        self.conv4_2_thermal = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_2_bn_thermal = nn.BatchNorm2d(512)
        self.conv4_3_thermal = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3_bn_thermal = nn.BatchNorm2d(512)
        """
        fusion
        """
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) #38->19

        self.conv5_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Replacements for FC6 and FC7 in VGG16
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # atrous convolution
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.conv4_feat = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_feat_bn = nn.BatchNorm2d(512)

        self.load_pretrained_layers()

    def forward(self, image_rgb,image_thermal):
        """
        RGB
        """
        out_rgb = F.relu(self.conv1_1_bn_rgb(self.conv1_1_rgb(image_rgb)))
        out_rgb = F.relu(self.conv1_2_bn_rgb(self.conv1_2_rgb(out_rgb)))
        out_rgb = self.pool1_rgb(out_rgb)

        out_rgb = F.relu(self.conv2_1_bn_rgb(self.conv2_1_rgb(out_rgb))) # (N, 128, 150, 150)
        out_rgb = F.relu(self.conv2_2_bn_rgb(self.conv2_2_rgb(out_rgb))) # (N, 128, 150, 150)
        out_rgb = self.pool2_rgb(out_rgb)  # (N, 128, 75, 75)

        out_rgb = F.relu(self.conv3_1_bn_rgb(self.conv3_1_rgb(out_rgb))) # (N, 256, 75, 75)
        out_rgb = F.relu(self.conv3_2_bn_rgb(self.conv3_2_rgb(out_rgb)))  # (N, 256, 75, 75)
        out_rgb = F.relu(self.conv3_3_bn_rgb(self.conv3_3_rgb(out_rgb)))  # (N, 256, 75, 75)
        out_rgb = self.pool3_rgb(out_rgb)  # (N, 256, 38, 38), it would have been 37 if not for ceil_mode = True

        out_rgb = F.relu(self.conv4_1_bn_rgb(self.conv4_1_rgb(out_rgb)))  # (N, 512, 38, 38)
        out_rgb = F.relu(self.conv4_2_bn_rgb(self.conv4_2_rgb(out_rgb)))  # (N, 512, 38, 38)
        out_rgb = F.relu(self.conv4_3_bn_rgb(self.conv4_3_rgb(out_rgb)))  # (N, 512, 38, 38)

        """
        Thermal
        """
        out_thermal = F.relu(self.conv1_1_bn_thermal(self.conv1_1_thermal(image_thermal)))
        out_thermal = F.relu(self.conv1_2_bn_thermal(self.conv1_2_thermal(out_thermal)))
        out_thermal=self.pool1_thermal(out_thermal)

        out_thermal = F.relu(self.conv2_1_bn_thermal(self.conv2_1_thermal(out_thermal)))
        out_thermal = F.relu(self.conv2_2_bn_thermal(self.conv2_2_thermal(out_thermal)))
        out_thermal = self.pool2_thermal(out_thermal)

        out_thermal = F.relu(self.conv3_1_bn_thermal(self.conv3_1_thermal(out_thermal)))
        out_thermal = F.relu(self.conv3_2_bn_thermal(self.conv3_2_thermal(out_thermal)))
        out_thermal = F.relu(self.conv3_3_bn_thermal(self.conv3_3_thermal(out_thermal)))  # (N, 256, 75, 75)
        out_thermal = self.pool3_thermal(out_thermal) 

        out_thermal = F.relu(self.conv4_1_bn_thermal(self.conv4_1_thermal(out_thermal)))  # (N, 512, 38, 38)
        out_thermal = F.relu(self.conv4_2_bn_thermal(self.conv4_2_thermal(out_thermal)))  # (N, 512, 38, 38)
        out_thermal = F.relu(self.conv4_3_bn_thermal(self.conv4_3_thermal(out_thermal))) # (N, 512, 38, 38)
   
        """
        Fusion
        """
        conv4_3_feats = torch.cat([out_rgb, out_thermal], dim=1)
        out=conv4_3_feats
        conv4_3_feats = F.relu(self.conv4_feat_bn(self.conv4_feat(conv4_3_feats)))
        

        out = self.pool4(out)

        out = F.relu(self.conv5_1_bn(self.conv5_1(out)))
        out = F.relu(self.conv5_2_bn(self.conv5_2(out)))
        out = F.relu(self.conv5_3_bn(self.conv5_3(out))) 
        out = self.pool5(out) 

        out = F.relu(self.conv6(out)) 

        conv7_feats = F.relu(self.conv7(out)) 

        return conv4_3_feats, conv7_feats

    def load_pretrained_layers(self):

        state_dict = self.state_dict() 
        param_names = list(state_dict.keys())

        pretrained_state_dict = torchvision.models.vgg16_bn(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        for i, param in enumerate(param_names):  # excluding conv6 and conv7 parameters
            if i==70:
              break
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]
            if i!=0 and i!=1:
              state_dict[param_names[i+70]] = state_dict[param]

        for i, param in enumerate(param_names[140:162]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i+70]]
        
        state_dict['conv1_1_thermal.weight'] = (pretrained_state_dict[pretrained_param_names[0]][:,0,:,:]+pretrained_state_dict[pretrained_param_names[0]][:,1,:,:]+pretrained_state_dict[pretrained_param_names[0]][:,2,:,:]).unsqueeze(1)/3
        state_dict['conv1_1_thermal.bias'] = pretrained_state_dict[pretrained_param_names[1]][:]

        state_dict['conv5_1.weight']=torch.cat((state_dict['conv5_1.weight'],state_dict['conv5_1.weight']),dim=1)

        #state_dict['conv4_feat']=torch.cat((state_dict['conv4_3_rgb.weight'],state_dict['conv4_3_rgb.weight']),dim=1)
        #state_dict['conv4_feat_bn']=torch.cat((state_dict['conv4_3_bn_rgb.weight'],state_dict['conv4_3_bn_rgb.weight']),dim=1)
    
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3) ->m배만큼 줄임
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)   

        self.load_state_dict(state_dict)
        print("\nLoaded base model.\n")


class AuxiliaryConvolutions(nn.Module):
    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        # Auxiliary/additional convolutions on top of the VGG base
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  # stride = 1, by default #in:1024, out:256
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        out = F.relu(self.conv8_1(conv7_feats))  # (N, 256, 19, 19) : batchsize, output, h, w
        out = F.relu(self.conv8_2(out))  # (N, 512, 10, 10)
        conv8_2_feats = out  # (N, 512, 10, 10)

        out = F.relu(self.conv9_1(out))  # (N, 128, 10, 10)
        out = F.relu(self.conv9_2(out))  # (N, 256, 5, 5)
        conv9_2_feats = out  # (N, 256, 5, 5)

        out = F.relu(self.conv10_1(out))  # (N, 128, 5, 5)
        out = F.relu(self.conv10_2(out))  # (N, 256, 3, 3)
        conv10_2_feats = out  # (N, 256, 3, 3)

        out = F.relu(self.conv11_1(out))  # (N, 128, 3, 3)
        conv11_2_feats = F.relu(self.conv11_2(out))  # (N, 256, 1, 1)

        # Higher-level feature maps
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats


class PredictionConvolutions(nn.Module):
    def __init__(self, n_classes):
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        n_boxes = {'conv4_3': 4,
                   'conv7': 6,
                   'conv8_2': 6,
                   'conv9_2': 6,
                   'conv10_2': 4,
                   'conv11_2': 4}

        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * 4, kernel_size=3, padding=1)

        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * n_classes, kernel_size=3, padding=1)
#MultiBoxLoss
        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):

        batch_size = conv4_3_feats.size(0)

        l_conv4_3= self.loc_conv4_3(conv4_3_feats)  # (N, 16, 38, 38)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 38, 38, 16), to match prior-box order (after .view())
        l_conv4_3= l_conv4_3.view(batch_size, -1, 4)  # (N, 5776, 4), there are a total 5776 boxes on this feature map
    

        l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 19, 19)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 24)
        l_conv7 = l_conv7.view(batch_size, -1, 4)  # (N, 2166, 4), there are a total 2116 boxes on this feature map

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 10, 10)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 600, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 5, 5)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # (N, 150, 4)

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 3, 3)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  # (N, 36, 4)

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, 1, 1)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  # (N, 4, 4)

        # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 4 * n_classes, 38, 38)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        c_conv4_3 = c_conv4_3.view(batch_size, -1,
                                   self.n_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map
 
        c_conv7 = self.cl_conv7(conv7_feats)  # (N, 6 * n_classes, 19, 19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * n_classes)
        c_conv7 = c_conv7.view(batch_size, -1,
                               self.n_classes)  # (N, 2166, n_classes), there are a total 2116 boxes on this feature map

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, 6 * n_classes, 10, 10)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * n_classes)
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)  # (N, 600, n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, 6 * n_classes, 5, 5)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)  # (N, 150, n_classes)

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  # (N, 4 * n_classes, 3, 3)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * n_classes)
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)  # (N, 36, n_classes)

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  # (N, 4 * n_classes, 1, 1)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)

        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)  # (N, 8732, 4)
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2],
                                   dim=1)  # (N, 8732, n_classes)

        return locs, classes_scores


class SSD300(nn.Module):
    def __init__(self, n_classes):
        super(SSD300, self).__init__()

        self.n_classes = n_classes
        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)

        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image_rgb,image_thermal):

        # Run VGG base network convolutions (lower level feature map generators)
        conv4_3_feats, conv7_feats = self.base(image_rgb,image_thermal)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        conv4_3_feats = conv4_3_feats/ norm  # (N, 512, 38, 38)
        conv4_3_feats = conv4_3_feats * self.rescale_factors  # (N, 512, 38, 38)


        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = \
            self.aux_convs(conv7_feats)  # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats)  # (N, 8732, 4), (N, 8732, n_classes)

        return locs, classes_scores

    def create_prior_boxes(self):
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]
                    
                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])

                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):

        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)
  
        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    suppress = torch.max(suppress, overlap[box] > max_overlap)

                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


class MultiBoxLoss(nn.Module):

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.SmoothL1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        #assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)

        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)
        # import pdb;pdb.set_trace()
        # First, find the loss for all priors
        
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive

        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss
