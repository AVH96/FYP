from torch.autograd import Variable
import cv2
import numpy as np
import torch

# Function to obtain heatmap
def get_heatmap(model,dataloaders):
    model.eval()
    img, _ = (next(iter(dataloaders['valid'])))
    inputs = Variable(img[0].cuda())
    # Obtain features maps of model
    _, features = model(inputs)
    # Obtain weights for feature maps
    out_fc = model.fc.weight
    out_fc = torch.squeeze(out_fc)
    out_fc[out_fc<0] = 0

    # normalise weights
    total = torch.sum(out_fc)
    out_fc = torch.div(out_fc,total)

    # multiply weighting to feature maps
    for j in range(len(out_fc)):
            features[:,j,:,:] *= out_fc[j]

    # Sum weighted maps together to get heatmap
    heatmap = torch.sum(features[2],dim=0)
    heatmap = heatmap.cpu().detach().numpy()
    heatmap = np.array(heatmap)

    # Load in X-ray image
    full_img = cv2.imread('D:/Desktop/FYP/MURA-vtest/valid/XR_WRIST/patient00006/study1_positive/image5')

    # normalise heat map and display on image
    heatmap = heatmap-np.min(heatmap)
    heatmap = heatmap/np.max(heatmap)
    heatmap = cv2.resize(heatmap,(full_img.shape[1], full_img.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + full_img*0.8
    cv2.imwrite('./map.jpg', superimposed_img)