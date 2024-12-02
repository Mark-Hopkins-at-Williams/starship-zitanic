import pandas as pd
import torch
from torch import tensor
from tqdm import tqdm
from torch.utils.data import Dataset

PLANETS = ['Earth', 'Europa', 'Mars', '???']
DESTINATIONS = ['55 Cancri e', 'PSO J318.5-22', 'TRAPPIST-1e', '???']


def read_csv(csv_file):
    """Reads a csv into a pandas dataframe."""
    data = pd.read_csv(csv_file)
    return data
  

def categorical_feature(row, feature_name, domain):
    """Creates binary features from a categorical feature."""    
    result = [0.0] * len(domain)
    result[domain.index(row[feature_name])] = 1.0    
    return tensor(result)


def create_features(row):
    """Creates a feature vector from a row of a dataframe."""    
    return torch.cat([
        categorical_feature(row, 'HomePlanet', PLANETS),
        categorical_feature(row, 'Destination', DESTINATIONS)
    ], dim=0)
 

class SpaceshipZitanicData(Dataset):
    def __init__(self, csv_file, test_set=False):
        data = read_csv(csv_file)
        self.x = []
        self.y = []
        for _, row in tqdm(data.iterrows()):              
            self.x.append({f: row[f] for f in set(data.columns) - {'Transported'}})
            if test_set:
                self.y.append(-1)
            else:
                self.y.append(row['Transported'])        
                      
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return (create_features(self.x[idx]), self.y[idx])  

    
if __name__ == "__main__":             
    train_set = SpaceshipZitanicData('data/train.csv')
    print(train_set[0])
    