import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'age': 45, 'gender':0,
	'polyuria':1, 'polydipsia':1, 'sudden_weight_loss':0,
    'weakness':0, 'polyphagia':0, 'genital_thrush':1, 
    'visual_blurring':1, 'itching':0, 'irritability':0,
    'delayed_healing':1, 'partial_paresis':1, 
    'muscle_stiffness':0, 'alopecia':1, 'obesity':0})

print(r.json())