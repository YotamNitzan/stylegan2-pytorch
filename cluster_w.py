from pathlib import Path
from argparse import ArgumentParser
import csv
from tqdm import tqdm

import torch
from torch import nn
from torchvision import datasets, models, transforms

import numpy as np
from PIL import Image


class Classifier(nn.Module):
    def __init__(self, device):
        super(Classifier, self).__init__()
        self.device = device
        model_ft = models.squeezenet1_0(pretrained=False)
        model_ft.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = 2
        model_ft.load_state_dict(torch.load('ffhq_and_afhq_cats_classifier.pt'))
        self.model = model_ft
        self.model.to(self.device)
        self.model.eval()
        self.input_size = 224
        self.transform = transforms.Compose([transforms.Resize(self.input_size),
                                             transforms.CenterCrop(self.input_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def forward(self, x):
        x = self.transform(x)
        x = x.to(self.device)
        x = torch.unsqueeze(x, 0)
        x = self.model(x)
        return x


def create_tsv_file(label_file: Path, data_dir: Path):
    with label_file.open('w') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        for w_file in tqdm(sorted(data_dir.glob('*.npy'))):
            w = np.load(w_file)[0]
            writer.writerow(w)


def label(label_file: Path, data_dir: Path, device):
    data = []
    model = Classifier(device)
    num_to_name = {0: 'cat', 1: 'person'}
    for w_file in tqdm(sorted(data_dir.glob('*.npy'))):
        img_file = w_file.with_suffix('.png')
        if not img_file.exists():
            print(f'Matching image file for {w_file} does not exist')
            continue

        img = Image.open(img_file)

        label = torch.argmax(model(img)).item()
        label = num_to_name[label]

        data.append([w_file.stem, label])

    with label_file.open('w') as csv_file:
        field_names = ['sample', 'label']
        writer = csv.writer(csv_file, delimiter='\t')

        writer.writerow(field_names)

        writer.writerows(data)


if __name__ == '__main__':
    parser = ArgumentParser(description='Cluster in W space')

    parser.add_argument('path', type=Path)
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()
    create_tsv_file(args.path.joinpath('data.tsv'), args.path)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    label(args.path.joinpath('meta-data.tsv'), args.path, device)
