# pip install flask
# pip install flask-cors
from flask import Flask, request, Response, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

def the_model(safe, abused, sexual_abused, teaching, grooming, identify, recovering, action):
	result_str = ''
	with open('logistic.pk', 'rb') as f:
		model = pickle.load(f)

		safe = 1 if safe == 'Agree' else 0
		abused = 1 if abused == 'Agree' else 0
		sexual_abused = 1 if sexual_abused == 'Agree' else 0
		teaching = 1 if teaching == 'Agree' else 0
		grooming = 1 if grooming == 'Yes' else 0
		identify = 1 if identify == 'Yes' else 0
		recovering = 1 if recovering == 'Yes' else 0
		action = 1 if action == 'Yes' else 0

		output = model.predict_proba([[safe, abused, sexual_abused, teaching, grooming, identify, recovering, action]])

		result_str += "Beginner " + "{:.4f}".format(output[0][0]) + "<br />"
		result_str += "Intermediate " + "{:.4f}".format(output[0][1]) + "<br />"

	return result_str

@app.route('/input', methods=['GET'])
def input():
	return render_template('input.html')

@app.route('/model-api', methods=['POST'])
def model_api():
	safe = request.form.get('safe')
	abused = request.form.get('abused')
	sexual_abused = request.form.get('sexual_abused')
	teaching = request.form.get('teaching')
	grooming = request.form.get('grooming')
	identify = request.form.get('identify')
	recovering = request.form.get('recovering')
	action = request.form.get('action')

	the_output = the_model(safe, abused, sexual_abused, teaching, grooming, identify, recovering, action)
	return the_output

if __name__ == '__main__':
	app.run(debug=True, port='8080', host='0.0.0.0', use_reloader=True)
