# pip install flask
# pip install flask-cors
from flask import Flask, request, Response, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

def the_model(age, sex, rurality, food, medication, sanitation):
	result_str = ''
	with open('logistic.pk', 'rb') as f:
		model = pickle.load(f)
		sex = 1 if sex == 'Female' else 0
		food = 1 if food == 'Food' else 0
		medication = 1 if medication == 'Medication' else 0
		sanitation = 1 if sanitation == 'Sanitation' else 0

		if (rurality == 'Suburban'):
			rurality = 1
		elif (rurality == 'Urban'):
			rurality = 2
		else:
			rurality = 0
		output = model.predict_proba([[age, sex, rurality, food, medication, sanitation]])

		result_str += "HUMAN " + "{:.2f}".format(output[0][0]) + "%" + "<br />"
		result_str += "ZOMBIE " + "{:.2f}".format(output[0][1]) + "%" + "<br />"
		return result_str

@app.route('/sample-url', methods=['GET'])
def sample_url():
	return render_template('sample-url.html')

@app.route('/input', methods=['GET'])
def input():
	return render_template('input.html')

@app.route('/model-api', methods=['POST'])
def model_api():
	age = request.form.get('age')
	sex = request.form.get('sex')
	rurality = request.form.get('rurality')
	food = request.form.get('food')
	medication = request.form.get('medication')
	sanitation = request.form.get('sanitation')

	age = int(age)

	the_output = the_model(age, sex, rurality, food, medication, sanitation)
	return the_output

if __name__ == '__main__':
	app.run(debug=True, port='8080', host='0.0.0.0', use_reloader=True)