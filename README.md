# Customer Churn Prediction
Below I offer an executive summary of how I developed the most accurate model on Kaggle's [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) dataset with a prediction accuracy of `91.30`. Please follow the markdown cells in my notebook to get an in depth look at my thought process.   

## Our Features
![the_features](images/the_features.png)  
## The Target
![the_targets](images/the_targets.png)  
## The libraries
Please run `conda env create -f environment.yml` from this repositories root directory to install all required libraries.  Summarizing, the primary ones are: `pytorch`, `imblearn`, `sklearn` & `pandas`.  

## The Model
![deeper_model](images/deeper_model_summary.png)

## The hyper-parameters
The loss function (criterion): `nn.BCEWithLogitsLoss()`.  
The optimizer and learning rate: `optim.Adam(deeper_model.parameters(), lr=1e-2)`.   
Validation set size: `20%`.  
The batch size: `827`.  
Instances of each class (after oversampling): `5174`.  
The number of epochs: `1000`. 

## Two custom functions
I wrote two custom functions `print_unique()` and `training_loop()`, both of which may be found in my notebook.

## "My Trick" to the most accurate model
If you've made it this far, take a peek inside my notebook. I'll give you a free cookie :) But, if you're in a rush, to summarize: The trick to achieving the highest accuracy out of any model on Kaggle, was to use`Adam` as my optimizer, which is more powerful than plain vanilla `SGD`, to synthetically oversample the minority class, and use a neural network with enough capacity, which happened to be two layers.