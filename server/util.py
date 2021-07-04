# importing module
import json
import pickle
import numpy as np


# Creating global variable
__locations=None # contains all the location
__data_columns=None # contain all the columns from the columns.json
__model = None # load the saved model

# prediction prices
def get_estimated_price(location,sqft,bhk,bath):
    '''
    creating numpy array with all zeroes
    we assign 1 only to the location which we want to predict price rest all are zeroes
    '''
    try:
        loc_index = __data_columns.index(location.lower()) # getting index of a particular location
    except:
        loc_index = -1
    x= np.zeros(len(__data_columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if loc_index >=0:
        x[loc_index]=1

    '''
    after model done the prediction we will get a 2d array back
    since our array have only one element we can access 0th location
    this will be a float value in lakh so we are rounding it upto two decimal places.
    '''
    return round(__model.predict([x])[0],2)



#getting all locations
def get_location_names():
    return __locations

#loading artifacts
def load_saved_artifacts():
    print("loading the saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    with open("./artifacts/columns.json",'r') as f:
        # loading all data columns form json
        __data_columns=json.load(f)['data_columns']
        # location names is starting from 3rd index so storing
        __locations = __data_columns[3:]
    # loading saved model
    with open("./artifacts/banglore_home_prices_model.pickle",'rb') as f:
        __model = pickle.load(f)



if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    # now predicting some sample values
    print(get_estimated_price('1st Phase JP Nagar',1000,3,3),'Lakh')
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2), 'Lakh')
    print(get_estimated_price('Kalhalli', 1000, 2, 2), 'Lakh') # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2), 'Lakh') # other location
