from flask import Flask, request, Response, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

def the_model(Car_Name, Year, Selling_Price, Kms_Driven, Fuel_Type, Transmission):
	result_str = ''
	with open('logistic.pk', 'rb') as f:
		model = pickle.load(f)
		Fuel_Type = 1 if Fuel_Type == 'Diesel' else 0
		Transmission = 1 if Transmission == 'Manual' else 0
		output = model.predict_proba([[Car_Name, Year, Selling_Price,Kms_Driven,Fuel_Type,Transmission]])
		if output[0][0] > output[0][1] and output[0][0] > output[0][2]:
			result_str += "First Owner is the highest percentage " + "{:.4f}".format(output[0][0]) + "<br />" + "<br />"
		if output[0][1] > output[0][0] and output[0][1] > output[0][2]:
			result_str += "Second Owner is the highest percentage " + "{:.4f}".format(output[0][0])
		if output[0][2] > output[0][0] and output[0][2] > output[0][1]:
			result_str += "Third Owner is the highest percentage " + "{:.4f}".format(output[0][0])
			
		result_str += "First Owner " + "{:.4f}".format(output[0][0]) + "<br />"
		result_str += "Second Owner " + "{:.4f}".format(output[0][1]) + "<br />"
		result_str += "Third Owner " + "{:.4f}".format(output[0][2]) + "<br />"

	return result_str

@app.route('/input', methods=['GET'])
def input():
	return render_template('input.html')

@app.route('/model-api', methods=['POST'])
def model_api():
	Car_Name = request.form.get('Car_Name')
	Year = request.form.get('Year')
	Selling_Price = request.form.get('Selling_Price')
	Kms_Driven = request.form.get('Kms_Driven')
	Fuel_Type = request.form.get('Fuel_Type')
	Transmission = request.form.get('Transmission')

	Car_Name = int(Car_Name)
	Year = int(Year)
	Selling_Price = int(Selling_Price)
	Kms_Driven = float(Kms_Driven)

	the_output = the_model(Car_Name, Year, Selling_Price, Kms_Driven, Fuel_Type, Transmission)
	return the_output

if __name__ == '__main__':
	app.run(debug=True, port='8080', host='0.0.0.0', use_reloader=True)