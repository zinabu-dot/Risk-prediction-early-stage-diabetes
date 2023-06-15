import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('diab_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    pred_prob = model.predict_proba(final_features)

    # Get the index of the predicted class for the first sample
    predicted_class_index = model.classes_.tolist().index(prediction[0])

    # Access the probability for the predicted class
    predicted_class_probability = pred_prob[0][predicted_class_index]
    

    output = round(prediction[0], 2)
    pred_proba = round(predicted_class_probability, 2)
    #"""
    if (output==0.00):
        return render_template('index.html', prediction_text='The predicted diabetes status is {}'.format(output)+', with prodicted probability {}'.format(pred_proba)+';'+ ' ' +'This means, based on the information supplied: No early diabetes symptoms. Do not forget to keep doing exercise and eating recommended diet.')
    else:
        return render_template('index.html', prediction_text='The predicted result indicated that it is class  {}'.format(output)+', with prodicted probability {}'.format(pred_proba)+';'+ ' ' +'perhaps, early diabetes symptoms. We would like to reiterate you that this is based on the information provided. Consult a practitionner, maintain physical exercise, and healthy diet.')
    #"""
    #return render_template('index.html', prediction_text='your diabetic status should be {}'.format(output))
        

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)