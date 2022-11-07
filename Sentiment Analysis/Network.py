#%% Importing
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from ToTensor import ToTensor
from TweetDataset_75 import VectorizedTweetsDataset
import time
from sklearn.metrics import confusion_matrix

#%% Defining Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(300, 150, 1, batch_first = True)
        self.linear = nn.Linear(150, 3)

    def forward(self, x, hidden, batch_size):
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, 150)
        x = self.linear(lstm_out)
        x = F.softmax(x)
        
        # reshape to be batch_size first
        x = x.view(batch_size, -1) 
        x = x[:, -4:-1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return x, hidden
    
    def init_hidden(self, batch_size):
        
        weight = next(self.parameters()).data
        
        hidden = (weight.new(1, batch_size, 150).zero_(),
                  weight.new(1, batch_size, 150).zero_())
        
        return hidden

#%% Training Step
train_loss_vector = []
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    h = model.init_hidden(args.batch_size)
    for batch_idx, sample in enumerate(train_loader):
        
        data = sample['tweet']
        target = sample['polarity']
        data, target = data.to(device), target.to(device)
        data = data.float()
        
        optimizer.zero_grad()
        
        h = tuple([each.data for each in h])
        output, h = model(data, h, args.batch_size)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            train_loss_vector.append(loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

#%% Testing Step
test_loss_vector = []
test_accuracy = []
def test(args, model, device, test_loader):
    model.eval()
    h = model.init_hidden(args.test_batch_size)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in test_loader:
            data = sample['tweet']
            target = sample['polarity']
            data, target = data.to(device), target.to(device)
            data = data.float()
            
            h = tuple([each.data for each in h])
            
            output, h = model(data, h, args.test_batch_size)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_loss_vector.append(test_loss)
    test_accuracy.append(100.*correct/len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(confusion_matrix(target.cpu(), pred.cpu()))

#%% Main
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch LSP')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=368, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    torch.manual_seed(args.seed)
    
    # https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    # train dataset
    train_set = VectorizedTweetsDataset('training.cleaned_short.csv',
                                        transform=transforms.Compose([ToTensor()]))
    
    # test dataset
    test_set = VectorizedTweetsDataset('testing.cleaned.csv',
                                       transform=transforms.Compose([ToTensor()]))

    # train data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=True, drop_last = True, **kwargs)

    # test data loader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size,
                                              shuffle=True, drop_last = True, **kwargs)

    # make a network instance
    model = Net().to(device)

    # configure optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # configure learning rate scheduler
#    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # train epochs
    ts = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
#        scheduler.step()
    te = time.time()
    duration = te - ts
    print(duration)

    # save the trained model
    if args.save_model:
	    torch.save(model.state_dict(), "alaki1.pt")

#%% Calling main
if __name__ == '__main__':
    main()
