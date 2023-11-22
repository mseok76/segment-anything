from predictor import SamPredictor
from build_sam import sam_model_registry
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch
import os
import argparse
from datetime import datetime

#Can use as parse : sam model version
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='../../../dataset/SAM_data_001/', help = 'make path from this train.py file')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run default = 30')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', default=0.0003, type=float, metavar ='N',
                    help='learning rate')

#parser.add_argument('--seed', default=None, type=int,
#                    help='seed for initializing training. ')
parser.add_argument("--save_path", type=str, default='./weights/')


def train(arg):
    now = datetime.now()

    # Loading the model     ***************need check**************
    sam = sam_model_registry["vit_l"](checkpoint=None)#"./models")    #conform checkpoint 
        #model type:h l b  & default is h
        #checkpoint is saved weight
    predictor = SamPredictor(sam)

    #save directory
    save_path = arg.save_path   #save as start time ./models/mdHM directory
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize to the size a model expects
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalization values for pre-trained PyTorch models
        #check normalize value
    ])

    # Load custom dataset  ********************need_chage*****mask_file_position****************
    dataset = CustomDataset(root_dir = arg.data_path + 'image/', mask_dir= arg.data_path + 'label/', transform=transform)  

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=arg.batch_size, shuffle=True)

    # Fine-tuning the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor.sam.to(device)
    predictor.sam.train()
    
    # Define loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(predictor.sam.parameters(), lr=arg.lr, momentum=0.9)

    
    for epoch in range(arg.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        num_batch = 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            outputs = predictor.sam(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batch = num_batch+1

        epoch_loss = running_loss/num_batch
        print('epoch : %d \t loss : %.3f\n'%(epoch,epoch_loss))

        if(epoch%10 == 0):
            arg.lr *= 0.9
            optimizer = torch.optim.Adam(sam.parameters(), lr=arg.lr)
            print("learning rate is decayed")

        if(epoch == 1):
            best_loss = epoch_loss
            
        elif(best_loss > epoch_loss):
            torch.save(sam.state_dict(), save_path + '/' + str(epoch)+ now.strftime('%m%d%H%M') + '.pkl')
            print("Save %d epoch value"%epoch+1)
        #highest result
    print('Finished Training')


# Define dataset with masks
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mask_dir, transform=None):
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.images[idx])
        mask_name = os.path.join(self.mask_dir, self.images[idx])
        
        image = Image.open(img_name)
        mask = Image.open(mask_name)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
    

if __name__=="__main__":
    arg = parser.parse_args()
    train(arg)
    