import torch
from torch.nn import Parameter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from features import SpaceshipZitanicData


class LogisticRegressionModel(torch.nn.Module):
    """A basic logistic regression model."""  
    def __init__(self, num_input_features):
        super().__init__()
        self.num_input_features=num_input_features
        self.theta = Parameter(torch.empty(1, num_input_features))        
        self.bias = Parameter(torch.zeros(1))
        for param in [self.theta]:
            torch.nn.init.kaiming_uniform_(param)
        
    def forward(self, x):
        result = self.theta @ x.t() + self.bias
        result = torch.sigmoid(result)
        return result.squeeze()
    

def nlog_prob_loss(model, x, y):
    """Computes the negative log probability of the correct response."""
    positive_response_probs = model.forward(x)
    correct_response_probs = torch.abs(1 - (positive_response_probs + y))
    losses = -torch.log(correct_response_probs)
    return torch.mean(losses)

        
def evaluate(model, test_loader):
    """Evaluates the model on a test set."""
    model.eval()
    correct = 0
    total = 0
    for x, y in test_loader:
        preds = (model.forward(x) > 0.5).long() 
        correct += torch.sum(preds == y).item()
        total += torch.numel(y)
    model.train()       
    return correct/total


def predict(model, test_loader):
    """Evaluates the model on a test set."""
    model.eval()
    predictions = []
    for x, _ in test_loader:        
        preds = (model.forward(x) > 0.5).long() 
        predictions.append(preds)
    return torch.cat(predictions, dim=0)


def plot_accuracies(accuracies):
    """Plots test accuracy as training proceeds."""
    sns.set_theme(style="darkgrid")
    plt.clf()
    sns.lineplot(x=list(range(len(accuracies))), y=accuracies)
    plt.xlabel("Training steps")
    plt.ylabel("Evaluation accuracy")
    plt.ion()
    plt.pause(0.01)
    

def gradient_descent(model, num_epochs, train_set, test_set, lr=0.01, plot_every=1):
    """Runs a basic version of gradient descent."""
    accuracies = []
    for _ in tqdm(range(num_epochs)):    
        model.train()
        train_loader = DataLoader(train_set, batch_size=128, drop_last=True)
        for x, y in train_loader:
            loss = nlog_prob_loss(model, x, y)
            loss.backward()
            for param in model.parameters():
                with torch.no_grad():           
                    param -= lr*param.grad
                    param.grad = None
        test_loader = DataLoader(test_set, batch_size=512, drop_last=True)
        accuracy = evaluate(model, test_loader)
        accuracies.append(accuracy)
        if len(accuracies) % plot_every == 0:
            plot_accuracies(accuracies)        
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    train_set = SpaceshipZitanicData('data/train.csv')
    dev_set = SpaceshipZitanicData('data/dev.csv')       
    num_epochs = 100
    x, y = train_set[0]
    num_features = x.shape[0]
    model = LogisticRegressionModel(num_features)
    trained_model = gradient_descent(model, num_epochs, train_set, dev_set)    
    
    test_set = SpaceshipZitanicData('data/test.csv', test_set=True)     
    test_loader = DataLoader(test_set, batch_size=512)          
    results = predict(model, test_loader)
    with open('predictions.txt', 'w') as writer:           
        for result in results:
            writer.write(str(result.item()) + '\n')
        