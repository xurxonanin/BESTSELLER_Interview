# House Price Prediction Application
This is the code created for an application that predicts the price per square unit of a house given 6 inputs: the date of the transaction, the house age, the distance to the closest public transport stop, the number of convenience stores close by and its coordinates in latitude and longitude.

## Prediction Model and Data Pipeline
The data is first cleaned to avoid NaNs in the inputs for the model and then the following transformations are done:
1. The date of the transaction is converted from decimal number into year-month-day in order for the model to take into account seasonability to check whether it affects the price. The day is fixed to the first day of the month since from the existing data it was not possible to extract the day in an accurate manner.
2. The rest of the parameters are scaled using standard score to avoid extreme values affecting the inference of the model.

For the model, the selected approach was a simple regressor using ElasticNet with alpha value 1.0 and l_ratio of 0.5, default on scikit-learn. It was trained using an splitted dataset using the 75% of the dataset keeping the same random state to have replicable results and using the same data in repeated experiments.

## API Development
The API was developed using FastAPI since it allows extended functionality such as type checking and data input validation while being quite straigthforward in the implementation. Two endpoints were created, the first "/health" is a GET call that returns an OK message if the system is operating normally.
The second one is a POST call to "/predict" that sends 6 parameters in a JSON string in the body of the request and fetches the data. The following constraints were included to validate the data:

1. "transaction_date": Must be a floating point value between 1900 and 2100.
2. "house_age": Must be a floating point value between 0 and 300.
3. "distance_to_MRT": Must be a floating point value between 0 and 50000.
4. "number_of_CS": Must be an integer value greater than 0 and 150.
5. "latitude": Must be a floating point value between 22 and 26.
6. "longitude": Must be a floating point value between 120 and 123.

## Tests
A series of 8 tests were created to check that the model is correctly loaded, that it returns the expected outputs, that the endpoints are available and how the POST call behaves depending on the format of the data sent.