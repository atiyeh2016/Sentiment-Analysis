#%% Importing
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from ToTensor import ToTensor
from TweetDatasetBert2 import VectorizedTweetsDataset
import time
from sklearn.metrics import confusion_matrix
#from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertModel

#%% Defining Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 3,
#                                                      output_attentions = False,output_hidden_states = True)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, 3)
        
    def forward(self, seq, attn_masks, batch_size):
        seq = seq.view(batch_size,-1)
        attn_masks = attn_masks.view(batch_size,-1)
        bert_out, _ = self.bert(seq, attention_mask = attn_masks)
        cls_rep = bert_out[:,0]
        x = self.linear(cls_rep)
        logits = F.softmax(x)
    
        return logits

#%% Training Step
train_loss_vector = []
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        
        data = sample['tweet']
        attn_masks = sample['attention_mask']
        target = sample['polarity']
        data, target, attn_masks = data.to(device), target.to(device), attn_masks.to(device)
        
        optimizer.zero_grad()
        
        logits = model(data, attn_masks, args.batch_size)
                          
        loss = nn.CrossEntropyLoss()(logits, target)
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
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in test_loader:
            data = sample['tweet']
            attn_masks = sample['attention_mask']
            target = sample['polarity']
            data, target, attn_masks = data.to(device), target.to(device), attn_masks.to(device)
                        
            logits = model(data, attn_masks, args.test_batch_size)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(logits, target).item() # sum up batch loss
            pred = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
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
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
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
    sentence_length = 300
    train_set = VectorizedTweetsDataset('training.cleaned.csv', sentence_length)
    
    # test dataset
    test_set = VectorizedTweetsDataset('testing.cleaned.csv', sentence_length)


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
	    torch.save(model.state_dict(), "Bert_1.pt")

#%% Calling main
if __name__ == '__main__':
    main()
