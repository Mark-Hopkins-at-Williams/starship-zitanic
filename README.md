# Spaceship Zitanic

![The Spaceship Zitanic](images/spaceship.png)

In the year 3000 AD, cruise vacations aboard interstellar spaceships have experienced a recent resurgence in popularity. 

Unfortunately, during the maiden voyage of the Spaceship Zitanic, an _incident_ occurred. Namely, there was a time-space anomaly that caused nearly half of the cruise ship passengers to be transported to an alternate dimension. But which ones?!

At Mission Control, we only know the status of some of the passengers. This information has been split into two files: ```data/train.csv``` and ```data/dev.csv```.

As for the remaining passengers, we don't know whether they were transported to an alternate dimension or not. To assist in our rescue efforts, we would like to use machine learning to predict which of these passengers were transported. Data for these passengers is provided in ```data/test.csv```.

**Your task:** make these predictions with high accuracy!

## Evaluation

There are 20 points available:
- **Show up to class on Monday and work diligently:** 5 points
- **Show up to class on Wednesday and work diligently:** 5 points
- **Show up to class on Friday for the fun denouement:** 4 points
- There are 6 more points available, based on the accuracy of your classifier:
  - **+2**: 70-74% test accuracy
  - **+3**: 75-79% test accuracy
  - **+4**: 80-84% test accuracy
  - **+5**: 85-89% test accuracy
  - **+6**: 90-100% test accuracy


## Features

The following data has been collected for most passengers:

- ```PassengerId```: A unique Id for each passenger. Each Id takes the form ```gggg_pp``` where ```gggg``` indicates a group the passenger is travelling with and ```pp``` is their number within the group.
- ```HomePlanet```: The planet the passenger departed from, typically their planet of permanent residence.
- ```CryoSleep```: Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
- ```Cabin```: The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
- ```Destination```: The planet the passenger will be debarking to.
- ```Age```: The age of the passenger.
- ```VIP```: Whether the passenger has paid for special VIP service during the voyage.
- ```RoomService```, ```FoodCourt```, ```ShoppingMall```, ```Spa```, ```VRDeck```: Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
- ```Name```: The first and last names of the passenger.
- ```Transported```: Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.


## Starter Code

Some starter code is available in this repository:

- ```explore.py```: gives you an entrypoint for analyzing and visualizing the data
- ```features.py```: some starter code for creating a torch.nn.Dataset from the raw data
- ```train.py```: some simple logistic regression training code


## Submission

Your final submission should provide a text file called ```predictions.txt```. The nth line of the file should be a single binary digit corresponding to your prediction (1 if they were transported, 0 if not) for the nth passenger in ```data/test.csv```. Since there are 879 passengers in ```data/test.csv```, this file should be comprised of 879 lines.

Currently, running ```python train.py``` will create a baseline version of ```predictions.txt``` with an accuracy somewhere in the range of 56% to 60%. 


## Note

This challenge is derived from Kaggle's Spaceship Titanic competition. However, it is a custom-built edition for this class.
