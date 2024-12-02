import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_numeric_feature(data, column):
    sns.boxplot(x='Transported', y=column, data=data)
    plt.show()
    
    
def plot_categorical_feature(data, column):
    crosstab = pd.crosstab(data['Transported'], data[column])
    sns.heatmap(crosstab, annot=True, cmap='Blues', fmt='d')
    plt.xlabel(column)
    plt.ylabel('Transported')
    plt.show()


def read_csv(csv_file):
    data = pd.read_csv(csv_file)
    return data
    

def most_common_values(data, column):
    values = data[column].unique()
    print(f'{column}:')
    for value in values:
        print(f'  {value}')    
    value_counts = dict()
    for i, row in data.iterrows():
        feat = row[column]
        if feat not in value_counts:
            value_counts[feat] = 0
        value_counts[feat] += 1
    popular_values = [key for key in value_counts if value_counts[key] > 5 and type(key) == str]
    print(sorted(popular_values))   
    
    
def describe_dataset(data):
    def fun_title(text):
        print("\n" + "*"*len(str(text)))
        print(f"{text}")
        print("" + "*"*len(str(text)) + "\n")
    fun_title("Description of the dataset")
    print(data.describe())
    fun_title("Dataset statistics")    
    data.info()
    fun_title("Column values")  
    for column in data.keys():
        values = data[column].unique()
        print(f'{column}: {len(values)}')
    
    
if __name__ == "__main__":
    data = read_csv('data/train.csv')     
    # describe the dataset
    #   describe_dataset(data)  
    # or plot a numeric feature against the response variable
    #   plot_numeric_feature(data, 'RoomService')
    # or plot a categorical feature against the response variable
    #   plot_categorical_feature(data, 'HomePlanet')
   
    