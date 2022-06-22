from flask import Flask, request, render_template
import numpy as np
from joblib import load


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict',methods=['POST','GET'])
def get_delay():
    if request.method=='POST':
        result=request.form
		
        #import model
        clf = load('model/my_model.joblib')
		
        #import data
        X = load('model/data.joblib')
        prediction = np.array([c.predict(X[int(result['start']) : int(result['end'])]) for c in
          clf])
        prediction = prediction.mean(axis=0)

        return render_template('result.html', prediction = prediction)

    
if __name__ == '__main__':
	app.run()