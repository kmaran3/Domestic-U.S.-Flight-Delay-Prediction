# Prediction of Flight Delays
4242 Project: Team 36
Team Members: Yigithan Akercan, Shaambhav Dave, Zack Dearman, Ronith Gonsalves, Anish Jaisinghani & Krishna Maran


## Description
In the fast world of air travel, flight delays and cancellations are a common frustration for passengers, impacting operational efficiency, economic production, and customer satisfaction. Understanding the root of these delays and being able to predict them consistently is critically important in the aviation industry, for airlines, airports, and travelers alike.

This package visualizes a random forest regressor algorithm for predicting continuous values of delay times, trained using data from the United States Bureau of Transportation Statistics from 2019 through 2023, that investigates the root cause of delays making travel less hassle for customers, airlines, and airports. On the webpage tool, users can input their Airport, Airline, and month of their upcoming flight to see the predicted flight delay time and cancellation percentage. Users can also hover over any of the top 63 busiest airports in the US to see the top 5 airlines that fly out of the airport with the lowest expected delay.


## Installation
1. Download the package as a zip folder. All data a necessary libraries are pre-included
2. Setup a local HTTP server to host the webpage visualization. This can be done by running the command below in your console
```bash
python -m http.server 8000
``` 
4. Navigate to the file location of the package and open projectVisual.html
5. Have fun predicting flight delays!


## Execution
There are two ways to use our tool, each serving its own purpose
1. You have already booked your flight and want to see if it is expected to be delayed: Select the airline, airport, and month of your flight using the dropdown menus and press the submit button. After pressing submit, the expected delay time and probability of cancellation appear below
2. You want to fly out of a city but have not chosen a particular airline: Hover over the airport you want to fly out of. A tooltip showing the 5 best airlines with the lowest delay times appears. These airlines have the lowest expected delay from the chosen city


## Additional Notes
1. This portion is for people who would like to continue developing the project. It is not necessary for running the visualizations. The code we provided is open to the public and "rfc_2features.py", "rfc_3features.py", "gbc_2features.py", and "gbc_3features.py" demonstrate how Gradient Boosting Classifier and Random Forest Classifier used on testing accuracies of delay prediction using AIRLINE_CODE, ORIGIN_CODE, and Month of Travel features.
2. Even though the processed dataset is provided for the visualization with AIRLINE_CODE, ORIGIN_CODE, and Month of Travel, if any need to implement arises for future development, viz2_data_clean_train.py is also provided to predict delay times using Random Forest Regressor. The link to the original dataset: https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023/download?datasetVersionNumber=7 
