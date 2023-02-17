from flask import Flask, request, Response, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

def the_model(id, diagnosis, radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean):
	result_str = ''
	with open('logistic.pk', 'rb') as f:
		model = pickle.load(f)
		diagnosis = 1 if diagnosis == 'M' else 0
		output = model.predict_proba([[id, diagnosis, radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean]])
		result_str += "Low Chance: " + "{:.4f}".format(output[0][0]) + "<br />"
		result_str += "High Chance: " + "{:.4f}".format(output[0][1]) + "<br />"

	return result_str

@app.route('/input', methods=['GET'])
def input():
	return render_template('input.html')

@app.route('/model-api', methods=['POST'])
def model_api():
	id = request.form.get('id')
	diagnosis = request.form.get('diagnosis')
	radius_mean = request.form.get('radius_mean')
	texture_mean = request.form.get('texture_mean')
	perimeter_mean = request.form.get('perimeter_mean')
	area_mean = request.form.get('area_mean')
	smoothness_mean = request.form.get('smoothness_mean')
	compactness_mean = request.form.get('compactness_mean')
	concavity_mean = request.form.get('concavity_mean')

	id	= int(id)
	radius_mean = float(radius_mean)
	texture_mean = float(texture_mean)
	perimeter_mean = float(perimeter_mean)
	area_mean = float(area_mean)
	smoothness_mean = float(smoothness_mean)
	compactness_mean = float(compactness_mean)
	concavity_mean = float(concavity_mean)

	the_output = the_model(id, diagnosis, radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean)
	return the_output

if __name__ == '__main__':
	app.run(debug=True, port='8080', host='0.0.0.0', use_reloader=True)