# -*- coding: utf-8 -*-
"""
AirBnb_pkg - Package created to contain all functions required to process Airbnb data

"""

##Imports

import pandas as pd
import numpy as np

##Sklearn for data prep

from sklearn.preprocessing import MultiLabelBinarizer
import re

##Initial cleaning functions


def clean_calendar_data(data):
    """Perform basic cleaning on raw calendar data
    
    data: dataFrame to be cleaned
    """
    
    #remove NaN
    data = data.dropna()
    
    #Update formatting of prices
    data['price'] = [int(str(x).replace('$', '').replace(',','').replace('.00',''))
                    for x in data['price']]
    
    #Set Date column to datetime
    data['date'] = pd.to_datetime(data['date'])
    
    #Drops extra columns
    data = data[['listing_id', 'price', 'date']]
    
    return data


def clean_listing_data(data):
    """Basic cleaning of raw listing data
    
    data: dataFrame to be cleaned    
    """
    
    #Keep only columns that might influence cost, drop others (dates of reviews, host info, etc)
    keep_col = ['id', 'neighbourhood', 'zipcode', 'property_type', 
                'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 
                'bed_type', 'amenities', 'square_feet', 'price', 'weekly_price', 
                'monthly_price', 'number_of_reviews', 'review_scores_rating', 
                'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 
                'review_scores_communication', 'review_scores_location', 'review_scores_value',
                'reviews_per_month']
    
    data = data[keep_col]
    
    #Removes columns with more than 90% of data missing & print column
    data_dpcol = data.dropna(axis=1, thresh=len(data)*0.1)
    dropped_col = set(data.columns).difference(list(data_dpcol.columns))
    print(f'Columns dropped: {dropped_col}')
    
    #Removes rows with all missing values
    data_fil = data_dpcol.dropna(axis=0, thresh=1)
    print(f'Number of rows dropped: {len(data_dpcol)-len(data_fil)}')
    
    #Formatting price columns
    data_fil['price'] = data_fil['price'].fillna(0)
    
    data_fil['price'] = [int(str(x).replace('$','').replace(',','').replace('.00','')) 
                        for x in data_fil['price']]
    
    return data_fil

def data_merge(calendar_data, listings_data):
    """This function merges the calendar and listings datesets
    
    calendar_data: dataframe with prices for each listing per date
    listings_data: listing information & metadata
    """
    
    #Group calendar data by listing
    calendar_data = calendar_data.groupby('listing_id').median().round()
    calendar_data.columns = ['Cal_price']
    
    #Set index for listing as id
    listings_data.index = listings_data['id']
    listings_data = listings_data.drop(columns = ['id'])
    
    merged_data = pd.merge(calendar_data, listings_data, left_index=True, right_index=True)
    
    return merged_data

def ML_preprocessing(calendar_data, listings_data):
    """This function prepares the data for ML via feature reduction,
    expansion of nested features, and one hot encoding
    
    calendar_data: dataframe with prices for each listing per date
    listings_data: listing information & metadata
    """
    
    #Initial clean up of data
    cal_cleaned = clean_calendar_data(calendar_data)
    list_cleaned = clean_listing_data(listings_data)
    
    #Merge calendar & Listing data
    data_merged = data_merge(cal_cleaned, list_cleaned)

    #Drop redundant features from listings
    data_drop = data_merged.drop(columns=['price', 'neighbourhood'])
    
    #Replace numerical category variables with mode
    numeric_cata = ['review_scores_rating','accommodates','bathrooms','bedrooms','beds','review_scores_accuracy',
                    'review_scores_cleanliness','review_scores_checkin','review_scores_communication',
                    'review_scores_location','review_scores_value']

    data_drop[numeric_cata] = data_drop[numeric_cata].fillna(data_drop.mode().iloc[0])
    
    #Edit bad zip codes
    data_drop['zipcode'] = [str(x)[:3] for x in data_drop['zipcode']]
    
    #Drop NaN's
    data_drop = data_drop.dropna()
    
    #Seperate anemities into list
    data_drop['amenities'] = [x.replace('{','').replace('}','').replace('"','').split(',')
                               for x in data_drop['amenities']]
    
    #One hot encode the amenities & drop meaningless features
    mlb = MultiLabelBinarizer()
    amenities = pd.DataFrame(mlb.fit_transform(data_drop.pop('amenities')), columns = mlb.classes_,
                         index=data_drop.index)
    
    #Specific to vancouver dataset
    amenities = amenities.drop('translation missing: en.hosting_amenity_49', 1)
    amenities = amenities.drop('translation missing: en.hosting_amenity_50', 1)
    
    #get dummies for this non-numerica data
    non_numeric_data = ['zipcode', 'property_type', 'room_type', 'bed_type']
    nn_cata = pd.get_dummies(data_drop[non_numeric_data])
    
    #Merge the dataframes and drop extra columns to create the ML input
    data_temp = pd.merge(data_drop, nn_cata, left_index=True, right_index=True)
    data_output = pd.merge(data_temp, amenities, left_index=True, right_index=True)
    data_output = data_output.drop(columns=['zipcode', 'property_type','room_type', 'bed_type'])
    
    
    return data_output