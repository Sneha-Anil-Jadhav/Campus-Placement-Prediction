import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open(r'model_pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 1)
    if prediction == 0:
        return render_template('index.html', prediction_text='SORRY YOU ARE NOT SELECTED.... TRY AGAIN'.format(output))
    else:
        return render_template('index.html', prediction_text='CONGRATULATIONS!!! YOU ARE SELECTED'.format(output))
        
   
    
    #return render_template('index.html', prediction_text='Your Campus Placement is: {}'.format(output))
    
    
        

if __name__ == "__main__":
    app.run(debug=True)
    
