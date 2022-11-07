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
from PRUTransforms import *
from torch.autograd import Variable

#%% Defining Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.nhid = 64
        self.ninp = 300
        self.nlayers = 1
        self.k = 2
        self.g = 1
        self.pt = PyramidalTransform(300, 64, k=self.k)
        self.glt = GroupedLinear(64, 1)
        self.decoder = nn.Linear(4,3)
        
    def PRU(self, input, hidden):
        def recurrence(input_, hx):
            """Recurrence helper."""
            h_0, c_0 = hx[0], hx[1]

            # input vector is processed by Pyramidal Transform
            i2h_f, i2h_g, i2h_i, i2h_o = self.pt(input_)
            # previous hidden state is processed by the Grouped Linear Transform
            h2h_f, h2h_g, h2h_i, h2h_o = self.glt(h_0)

            # input to LSTM gates
            f = i2h_f + h2h_f
            g = i2h_g + h2h_g
            i = i2h_i + h2h_i
            o = i2h_o + h2h_o

            # outputs
            c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)
            return h_1, c_1
        
        
        input = input.transpose(0, 1)# batch is always first
        output = []
        steps = range(input.size(1))
        for i in steps:
            size_inp = input[:,i].size()
            input_t = input[:, i].view(size_inp[0], 1, size_inp[1]) # make input as
            hidden = recurrence(input_t, hidden)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

        output = torch.stack(output, 1) # stack the all output tensors, so that dims is 1 X Se X B X D
        output = torch.squeeze(output, 0) #remove the first dummy dim
        return output, hidden
     
    def forward(self, input, hidden):
        raw_output = input
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l in range(self.k):
            raw_output, new_h = self.PRU(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != 1 - 1:
                outputs.append(raw_output)
        hidden = new_hidden
        output = raw_output
        outputs.append(output)
        
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        model_output = decoded.view(output.size(0), output.size(1), decoded.size(1))
        return model_output, hidden
        
    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)
    
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhid)).zero_(),
                    Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhid)).zero_())
                    for l in range(self.nlayers)]

#%% repackage_hidden
def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

#%% Training Step
train_loss_vector = []
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    hidden = model.init_hidden(args.batch_size)
    for batch_idx, sample in enumerate(train_loader):
        
        data = sample['tweet']
        target = sample['polarity']
        data, target = data.to(device), target.to(device)
        data = data.float()
        
        optimizer.zero_grad()
        
#        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
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
    parser.add_argument('--no-cuda', action='store_true', default=True,
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
	    torch.save(model.state_dict(), "pyramid.pt")

#%% Calling main
if __name__ == '__main__':
    main()
