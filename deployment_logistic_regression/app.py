# pip install flask
# pip install flask-cors
from flask import Flask, request, Response, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

def the_model(temperature, body_aches, runny_nose, num_people_at_home, working_at_home, covid_vaccinated):
	result_str = ''
	with open('logistic.pk', 'rb') as f:
		model = pickle.load(f)
		body_aches = 1 if body_aches == 'yes' else 0
		runny_nose = 1 if runny_nose == 'yes' else 0
		working_at_home = 1 if runny_nose == 'yes' else 0
		covid_vaccinated = 1 if runny_nose == 'yes' else 0
		output = model.predict_proba([[temperature, body_aches, runny_nose,num_people_at_home,working_at_home,covid_vaccinated]])
		result_str += "FLU " + "{:.4f}".format(output[0][0]) + "<br />"
		result_str += "COVID " + "{:.4f}".format(output[0][1]) + "<br />"
		result_str += "COLDS " + "{:.4f}".format(output[0][2]) + "<br />"
	return result_str

@app.route('/sample-url', methods=['GET'])
def sample_url():
	return render_template('sample-url.html')

@app.route('/input', methods=['GET'])
def input():
	return render_template('input.html')

@app.route('/model-api', methods=['POST'])
def model_api():
	temperature = request.form.get('temperature')
	body_aches = request.form.get('body_aches')
	runny_nose = request.form.get('runny_nose')

	num_people_at_home = request.form.get('num_people_at_home')
	working_at_home = request.form.get('working_at_home')
	covid_vaccinated = request.form.get('covid_vaccinated')
	temperature	= float(temperature)
	num_people_at_home = int(num_people_at_home)
	the_output = the_model(temperature, body_aches, runny_nose, num_people_at_home, working_at_home, covid_vaccinated)
	return the_output

if __name__ == '__main__':
	app.run(debug=True, port='8080', host='0.0.0.0', use_reloader=True)