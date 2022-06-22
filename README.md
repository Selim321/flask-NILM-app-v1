# flask-NILM-app-v1
A flask app that predicts the state of each appliance in a given time period. The app take as input time interval and returns a vector that represents the predicted states of each appliance from the aggregate power data persisted in model/data.joblib. The considered appliances are: fridge and washing machine.   

## Notes
To run the app locally on a windows machine:
  1. Clone the repo.
  2. Create a virtual environment and install the packages in the requirements.txt.
  3. Persiste the model by using the model.py and then uncomment the line where the model is converted to a .joblib file.
  4. Run a local server by opening a terminal and tapping : "set FLASK_APP = app" and on a new line "python app.py". 
  5. Copy the URL and open it in your browser. 
