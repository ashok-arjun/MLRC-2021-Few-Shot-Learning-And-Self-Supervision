import torch
import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.transforms import Resize 
from resnet_pytorch import *
import json
import argparse
import string

from sklearn.linear_model import LogisticRegression

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
# parser.add_argument("--positive_json", "-p", type=str, default="test")
parser.add_argument("--positive_json", "-p", type=str, required=True)
parser.add_argument("--save_path", "-s", type=str, required=True)

args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed) 

class IterDataset(Dataset):
    def __init__(self, filelist):
        self.filelist = filelist

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])


    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        fn = os.path.join(self.filelist[idx])
        image = Image.open(fn).convert('RGB')        
        image = self.transforms(image)
        return image, fn

def get_features(model, dir=None, num_images=0, batch_size=64, num_workers=12, req_files=None):

    if req_files is None:
        if type(dir) == list:
            files = []
            for x in dir:
                x_files = list(os.listdir(x))
                x_files = [os.path.join(x, file) for file in x_files]
                files.extend(x_files)
        else:
            files = list(os.listdir(dir))
            files = [os.path.join(x, file) for file in files]

        req_files = random.sample(files, num_images)

    dataset = IterDataset(req_files)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)

    features = None
    filenames = []
    with torch.no_grad():
        for i, (img, fns) in enumerate(train_loader):
            img = img.cuda()
            feats = model.forward(img, features=True)

            if features is not None:
                features = torch.cat((features, feats))
            else:
                features = feats

            filenames.extend(fns)
            del img, feats

    return features, filenames
    
# Setup

set_seed(42)
torch.cuda.set_device(0)

# Model

model = resnet101(pretrained=True)
model = model.cuda()

# Open json and read image_names

file = open(args.positive_json)
dictionary = json.load(file)
features_pos, filenames_pos = get_features(model, req_files=dictionary["image_names"])

# Negative, naive reading
features_neg, filenames_neg = get_features(model, ["filelists/open-images/validation", "filelists/inat/images"], 10 * features_pos.shape[0])

print("Loaded")


#Training
print("Starting training...")

features_pos = features_pos.cpu().numpy()
labels_pos = np.repeat(1, features_pos.shape[0])
filenames_pos = np.array(filenames_pos)


features_neg = features_neg.cpu().numpy()
labels_neg = np.repeat(0, features_neg.shape[0])
filenames_neg = np.array(filenames_neg)

features = np.concatenate((features_pos, features_neg))
labels = np.concatenate((labels_pos, labels_neg))

clf = LogisticRegression(C=0.01, random_state=0, solver="lbfgs", multi_class="ovr", max_iter=1000, verbose=True, class_weight='balanced').fit(features, labels)


print("Done.")


probabilites = clf.predict_proba(features_neg) # can do in for loop to avoid memory drain
ratios = [num[1]/num[0] for num in probabilites]
ratios = np.array(ratios)

print("Total positive images: ", features_pos.shape[0])
print("Selecting top-%d negative images for SSL" % int(0.8 * features_pos.shape[0]))

required_num_images = int(0.8 * features_pos.shape[0])

sorted_ratios_indices = np.argsort(ratios)[::-1][:required_num_images]

sorted_ratios_indices = sorted_ratios_indices.astype(int)

req_features = features_neg[sorted_ratios_indices]
req_filenames = [filenames_neg[i] for i in sorted_ratios_indices]
req_labels = labels_neg[sorted_ratios_indices] # will be all 0

dictionary = {"image_names": req_filenames, "image_labels": req_labels.tolist()}

out_file = open("%s" % (args.save_path), "w")     
json.dump(dictionary, out_file, indent = 6)     
out_file.close() 

print("Saved %s" % (args.save_path))


