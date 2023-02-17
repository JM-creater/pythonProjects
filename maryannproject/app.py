from flask import Flask, request, Response, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

def the_model(ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes):
	result_str = ''
	with open('logistic.pk', 'rb') as f:
		model = pickle.load(f)
		output = model.predict_proba([[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes]])
		result_str += "Not Potable: " + "{:.4f}".format(output[0][0]) + "<br />"
		result_str += "Potable: " + "{:.4f}".format(output[0][1]) + "<br />"
	return result_str

@app.route('/input', methods=['GET'])
def input():
	return render_template('input.html')

@app.route('/model-api', methods=['POST'])
def model_api():
	ph = request.form.get('ph')
	Hardness = request.form.get('Hardness')
	Solids = request.form.get('Solids')
	Chloramines = request.form.get('Chloramines')
	Sulfate = request.form.get('Sulfate')
	Conductivity = request.form.get('Conductivity')
	Organic_carbon = request.form.get('Organic_carbon')
	Trihalomethanes = request.form.get('Trihalomethanes')

	ph	= float(ph)
	Hardness = float(Hardness)
	Solids = float(Solids)
	Chloramines = float(Chloramines)
	Sulfate = float(Sulfate)
	Conductivity = float(Conductivity)
	Organic_carbon = float(Organic_carbon)
	Trihalomethanes = float(Trihalomethanes)

	the_output = the_model(ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes)
	return the_output

if __name__ == '__main__':
	app.run(debug=True, port='8080', host='0.0.0.0', use_reloader=True)