import os
import torch
from PIL import Image
import numpy as np
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms

import transforms as ext_transforms
from models.pspnet import PSPNet
from data.utils import get_files
import argparse
import utils

parser = argparse.ArgumentParser(description='PSPNet Model eval')
parser.add_argument("--dataset-dir", type=str, default="../dataset/cityscape", help="Path to the root directory of the selected dataset. "
                    "Default: ../dataset/cityscapes")
parser.add_argument("--test_folder", type=str, default="leftImg8bit/test", help="Path of the selected test data. "
                    "Default: leftImg8bit/test")
parser.add_argument("--model_path", type=str, default="./save/PSPNet_cityscapes/PSPNet.pth.tar", help="Path of the best results of model. "
                    "Default: ./save/PSPNet_cityscapes/PSPNet.pth.tar")
parser.add_argument("--height", type=int, default=360, help="The image height. Default: 360")
parser.add_argument("--width", type=int, default=480, help="The image height. Default: 480")
parser.add_argument("--visual", type=int, default=1, help="The image height. Default: True")
parser.add_argument('-o', '--output_img_dir', default='./results/images', type=str,
                    help='output dir for drawn images')

args = parser.parse_args()

color_encoding = OrderedDict([
            ('unlabeled', (0, 0, 0)),
            ('road', (128, 64, 128)),
            ('sidewalk', (244, 35, 232)),
            ('building', (70, 70, 70)),
            ('wall', (102, 102, 156)),
            ('fence', (190, 153, 153)),
            ('pole', (153, 153, 153)),
            ('traffic_light', (250, 170, 30)),
            ('traffic_sign', (220, 220, 0)),
            ('vegetation', (107, 142, 35)),
            ('terrain', (152, 251, 152)),
            ('sky', (70, 130, 180)),
            ('person', (220, 20, 60)),
            ('rider', (255, 0, 0)),
            ('car', (0, 0, 142)),
            ('truck', (0, 0, 70)),
            ('bus', (0, 60, 100)),
            ('train', (0, 80, 100)),
            ('motorcycle', (0, 0, 230)),
            ('bicycle', (119, 11, 32))
    ])

image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])

label_to_rgb = transforms.Compose([
    ext_transforms.LongTensorToRGBPIL(color_encoding),
    transforms.ToTensor()
])


def predict(model, image, device):
    image = image.to(device)
    model = model.to(device)
    model.eval()
    while torch.no_grad():
        # Make predictions!
        prediction, _ = model(image)

        # Predictions is one-hot encoded with "num_classes" channels.
        # Convert it to a single int using the indices where the maximum (1) occurs
        _, predictions = torch.max(prediction.data, 1)

        return predictions

if __name__ == '__main__':
    if torch.cuda.is_available():
        if args.cuda:
            device = 'cuda'
            torch.cuda.empty_cache()
        else:
            device = 'cpu'
    else:
        device = 'cpu'

    test_data = get_files(os.path.join(args.dataset_dir, args.test_folder), extension_filter='.png')

    #initialize the model and load the checkpoint of best model.
    num_classes = len(color_encoding)
    model = PSPNet(num_classes)
    checkpoint = torch.load(args.model_path)
    model = model.load_state_dict(checkpoint['state_dict'])

    output_img_dir = args.output_img_dir
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)

    for data_path in test_data:
        image_name =  data_path.split('/')[-1]
        data = Image.open(data_path)
        h, w, _ = np.array(data).shape
        image = image_transform(data)

        prediction = predict(model, image, device=device)

        save_png = transforms.Resize(w, h)(torch.ByteTensor(prediction))
        save_png = torchvision.utils.make_grid(save_png).numpy()

        Image.fromarray(save_png).save(os.path.join(output_img_dir, 'submission', image_name))

        if args.visual:
            color_prediction = utils.batch_transform(prediction.cpu(), label_to_rgb)
            utils.imshow_batch(image.data.cpu(), color_prediction, os.path.join(output_img_dir, 'visual', image_name))
