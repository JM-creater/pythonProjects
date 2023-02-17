from flask import Flask, request, Response, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

def the_model(age, sex, chest_pain_type, resting_bp, chol_in_mg, fasting_blood_sugar, restecg, maxheart_rate_achieved):
	result_str = ''
	with open('logistic.pk', 'rb') as f:
		model = pickle.load(f)
		sex = 1 if sex == 'm' else 0
		output = model.predict_proba([[age, sex, chest_pain_type, resting_bp, chol_in_mg, fasting_blood_sugar, restecg, maxheart_rate_achieved]])
		result_str += "Low Chance of Having Heart Failure: " + "{:.4f}".format(output[0][0]) + "<br />"
		result_str += "High Chance of Having Heart Failure: " + "{:.4f}".format(output[0][1]) + "<br />"
	return result_str

@app.route('/input', methods=['GET'])
def input():
	return render_template('input.html')

@app.route('/model-api', methods=['POST'])
def model_api():
	age = request.form.get('age')
	sex = request.form.get('sex')
	chest_pain_type = request.form.get('chest_pain_type')
	resting_bp = request.form.get('resting_bp')
	chol_in_mg = request.form.get('chol_in_mg')
	fasting_blood_sugar = request.form.get('fasting_blood_sugar')
	restecg = request.form.get('restecg')
	maxheart_rate_achieved = request.form.get('maxheart_rate_achieved')

	age	= int(age)
	chest_pain_type = int(chest_pain_type)
	resting_bp = int(resting_bp)
	chol_in_mg = int(chol_in_mg)
	fasting_blood_sugar = int(fasting_blood_sugar)
	restecg = int(restecg)
	maxheart_rate_achieved = int(maxheart_rate_achieved)

	the_output = the_model(age, sex, chest_pain_type, resting_bp, chol_in_mg, fasting_blood_sugar, restecg, maxheart_rate_achieved)
	return the_output

if __name__ == '__main__':
	app.run(debug=True, port='8080', host='0.0.0.0', use_reloader=True)