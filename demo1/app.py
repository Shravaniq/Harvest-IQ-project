




import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd


# Create flask app
flask_app = Flask(__name__)
import pickle

# Load the model from the file
try:
    with open("recomendation_model.pkl", "rb") as f:
        model1 = pickle.load(f)
except FileNotFoundError:
    print("File not found. Please check the file path.")
    # Handle the error accordingly
yield_model=pickle.load(open("yield_model.pkl", "rb"))
# ferti = pickle.load(open('fertilizer.pkl','rb'))
# model1 = pickle.load(open('classifier.pkl','rb'))

model= pickle.load(open('classifier.pkl', 'rb'))
encoder = pickle.load(open('fertilizer.pkl', 'rb'))

with open("processor_encode.pkl", "rb") as f:
    encode = pickle.load(f)

with open("rf_yield_model.pkl", "rb") as f:
    yield_model = pickle.load(f)




crop_names = ['grapes', 'muskmelon', 'coconut', 'chickpea', 'watermelon', 'papaya', 'pomegranate', 'pigeonpeas', 'apple', 'maize', 'blackgram', 'cotton', 'mothbeans', 'coffee', 'jute', 'orange', 'rice', 'banana', 'mungbean', 'kidneybeans', 'mango', 'lentil']
crops = ['Arecanut', 'Other Kharif pulses', 'Rice', 'Banana', 'Cashewnut', 'Coconut', 'Dry ginger', 'Sugarcane', 'Sweet potato', 'Tapioca', 'Black pepper', 'Dry chillies', 'other oilseeds', 'Turmeric', 'Maize', 'Moong(Green Gram)', 'Urad', 'Arhar/Tur', 'Groundnut', 'Sunflower', 'Bajra', 'Castor seed', 'Cotton(lint)', 'Horse-gram', 'Jowar', 'Korra', 'Ragi', 'Tobacco', 'Gram', 'Wheat', 'Masoor', 'Sesamum', 'Linseed', 'Safflower', 'Onion', 'other misc. pulses', 'Samai', 'Small millets', 'Coriander', 'Potato', 'Other Rabi pulses', 'Soyabean', 'Beans & Mutter(Vegetable)', 'Bhindi', 'Brinjal', 'Citrus Fruit', 'Cucumber', 'Grapes', 'Mango', 'Orange', 'other fibres', 'Other Fresh Fruits', 'Other Vegetables', 'Papaya', 'Pome Fruit', 'Tomato', 'Mesta', 'Cowpea(Lobia)', 'Lemon', 'Pome Granet', 'Sapota', 'Cabbage', 'Rapeseed &Mustard', 'Peas(vegetable)', 'Niger seed', 'Bottle Gourd', 'Varagu', 'Garlic', 'Ginger', 'Oilseeds total', 'Pulses total', 'Jute', 'Peas & beans (Pulses)', 'Blackgram', 'Paddy', 'Pineapple', 'Barley', 'Sannhamp', 'Khesari', 'Guar seed', 'Moth', 'Other Cereals & Millets', 'Cond-spcs other', 'Turnip', 'Carrot', 'Redish', 'Arcanut(Processed)', 'Atcanut(Raw)', 'Cashewnut Processed', 'Cashewnut Raw', 'Cardamom', 'Rubber', 'Bitter Gourd', 'Drum Stick', 'Jack Fruit', 'Snak Guard', 'Tea', 'Coffee', 'Cauliflower', 'Other Citrus Fruit', 'Water Melon', 'Total foodgrain', 'Kapas', 'Colocosia', 'Lentil', 'Bean', 'Jobster', 'Perilla', 'Rajmash Kholar', 'Ricebean (nagadal)', 'Ash Gourd', 'Beet Root', 'Lab-Lab', 'Ribed Guard', 'Yam', 'Pump Kin', 'Apple', 'Peach', 'Pear', 'Plums', 'Litchi', 'Ber', 'Other Dry Fruit', 'Jute & mesta']

# Sort the array alphabetically
sorted_crop_names = sorted(crop_names)



@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/crop_recommendation", methods=['GET','POST'])
def crop_recommendation():
    if request.method == 'POST':
        Nitrogen_val= float(request.form['Nitrogen'])
        Phosphorous_val = float(request.form['Phosphorous'])
        Pottasium_val = float(request.form['Pottasium'])
        Humidity_val = float(request.form['Humidity'])
        Temperature_val = float(request.form['Temperature'])
        ph_val = float(request.form['ph'])
        Rainfall_val = float(request.form['Rainfall'])


        values_list = [Nitrogen_val, Phosphorous_val, Pottasium_val, Humidity_val, Temperature_val, ph_val, Rainfall_val]
        values_array = np.array(values_list)
        values_array = np.array(values_list).reshape(1, -1)

        prediction = model1.predict(values_array)[0]

        # crop_features = [float(x) for x in request.form.values()]
        # features = [np.array(crop_features)]
        # prediction = model.predict(features)[0]  # Accessing the first (and only) prediction

        # Find the index of the non-zero element in the prediction array
        predicted_crop_index = np.argmax(prediction)

        # Map the predicted index to the corresponding crop name
        predicted_crop_name = sorted_crop_names[predicted_crop_index]
        return render_template('result.html', result=predicted_crop_name)

        # return render_template("crop.html", prediction_text="The crop recommended is {}".format(predicted_crop_name))
    # else:
    return render_template("crop.html")


@flask_app.route("/yield_prediction", methods=['GET','POST'])
def yield_prediction():
    if request.method == 'POST':
        fertilizer_val= float(request.form['fertilizer'])
        rainfall_val = float(request.form['rainfall'])
        season_val = request.form['season']
        state_val = request.form['state']
        crop_val = request.form['crop']
       
        features = np.array([[crop_val, season_val,state_val,rainfall_val, fertilizer_val,]],dtype=object)
        transformed_features = encode.transform(features)
        prediction2 = yield_model.predict(transformed_features).reshape(1,-1)

        return render_template('yield_result.html', result2=prediction2)

        # return render_template("crop.html", prediction_text="The crop recommended is {}".format(predicted_crop_name))
    # else:
    return render_template("Yield_Prediction.html")


@flask_app.route('/predict_fertilizer', methods=['GET', 'POST'])
def predict_fertilizer():
    if request.method == 'POST':
        # Extracting values from the form
        temp = float(request.form.get('temp'))
        humi = float(request.form.get('humid'))
        mois = float(request.form.get('mois'))
        soil = int(request.form.get('soil'))
        crop = int(request.form.get('crop'))
        nitro = float(request.form.get('nitro'))
        pota = float(request.form.get('pota'))
        phosp = float(request.form.get('phos'))
        
        # Prepare input data
        input_data = np.array([[temp, humi, mois, soil, crop, nitro, pota, phosp]])
        
        # Make prediction
        predicted_class = model.predict(input_data)
        
        # Decode predicted class
        predicted_fertilizer = encoder.inverse_transform(predicted_class)[0]
        
        return render_template('Fertilizer.html', x='Predicted Fertilizer is {}'.format(predicted_fertilizer))
    return render_template('Fertilizer.html', x='')


if __name__ == "__main__":
    flask_app.run(debug=True)
