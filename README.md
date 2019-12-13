# Airbnb-Market-Analysis
The aim of this project is to predict Airbnb price listing based on season, location and property features.

In this repository the CRSIP-DM data framework is used to analyze data for the Vancouver, BC, Canada Airbnb market. Analysis is split into initial exploration, data processing, and modelling/evaluation. The business driver to this project is to ensure market-driven pricing of individual owners so as to not loose potential revenue from undervaluation or booking losses from overvaluation.

Specific questions are:

  1. Does the price vary based on season?
  2. Which Airbnb listing attributes/features have the biggest impact on price?
  3. Using the above, can the price be accuracly predicted for an Airbnb using only latent data?
  
Cleaning and general functions are stored in the Airbnb_pkg

# Results
  1. Does the price vary based on season?

Yes, there is a clear evidence of a dramatically lower price in the month of September. During this time, Vancouver may see reduced tourist due to 1) unfavorable local weather, 2) start of academic calendar years. This may be useful data for persons looking to travel to the city during times or reduced tourist traffic.

  2. Which listing/attributes have the biggest impact on price?

The largest impact is occupancy potential, number of bedrooms, total reviews and reviews/month. This suggests that the largest latent feature to drive pricing upwards is the total number of guests which can be accomodated. This makes intuitive sense as a larger Airbnb which can hold more guests would be expected to be more expensive. The other high impacts features correlate with how popular a property is. This may lead a potential Airbnb host to incentivise reviews for their property in order to increase price and further demand.

  3. Can price be predicted using only latent data?
  
Removing features related to reviews, the price can be predicted quite well. Outlier property with high valuation are more difficult to predict. This model is accurate enough (75%) to recommend pricing to users, which fulfills the initial business questions.

# Acknowledgements

Data was sourced from http://insideairbnb.com/vancouver/.
  
