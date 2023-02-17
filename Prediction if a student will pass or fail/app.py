# pip install flask
# pip install flask-cors
from flask import Flask, request, Response, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

def the_model(student_number, quizzes, performance_task, activities, projects, submit_on_time, submit_late, paid):
	result_str = ''
	with open('logistic.pk', 'rb') as f:
		model = pickle.load(f)
		quizzes = 1 if quizzes == 'yes' else 0
		performance_task = 1 if performance_task == 'yes' else 0
		activities = 1 if activities == 'yes' else 0
		projects = 1 if projects == 'yes' else 0
		submit_on_time = 1 if submit_on_time == 'yes' else 0
		submit_late = 1 if submit_late == 'yes' else 0
		paid = 1 if paid == 'yes' else 0
		output = model.predict_proba([[student_number, quizzes, performance_task, activities, projects, submit_on_time, submit_late, paid]])
		result_str += "FAILED " + "{:.4f}".format(output[0][0]) + "<br />"
		result_str += "PASSED " + "{:.4f}".format(output[0][1]) + "<br />"
	return result_str

@app.route('/sample-url', methods=['GET'])
def sample_url():
	return render_template('sample-url.html')

@app.route('/input', methods=['GET'])
def input():
	return render_template('input.html')

@app.route('/model-api', methods=['POST'])
def model_api():
	student_number = request.form.get('student_number')
	quizzes = request.form.get('quizzes')
	performance_task = request.form.get('performance_task')
	activities = request.form.get('activities')
	projects = request.form.get('projects')
	submit_on_time = request.form.get('submit_on_time')
	submit_late = request.form.get('submit_late')
	paid = request.form.get('paid')

	student_number = int(student_number)

	the_output = the_model(student_number, quizzes, performance_task, activities, projects, submit_on_time, submit_late, paid)
	return the_output

if __name__ == '__main__':
	app.run(debug=True, port='8080', host='0.0.0.0', use_reloader=True)