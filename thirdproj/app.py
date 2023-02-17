from flask import Flask, request, Response, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

def the_model(employee_id, department, region, education, gender, recruitment_channel, no_of_trainings, age, previous_year_rating, length_of_service, awards_won, avg_training_score):
    result_str = ''
    with open('logistic.pk', 'rb') as f:
        model = pickle.load(f)

        if department == 'Operations':
            department = 1
        elif department == 'Technology':
            department = 2
        elif department == 'Analytics':
            department = 3
        elif department == 'R&D':
            department = 4
        elif department == 'Procurement':
            department = 5
        elif department == 'Finance':
            department = 6
        elif department == 'HR':
            department = 7
        elif department == 'Legal':
            department = 8
        else:
            department = 0

        education = 1 if education == 'Master' else 0
        gender = 1 if gender == 'f' else 0
        recruitment_channel = 1 if recruitment_channel == 'other' else 0

        output = model.predict_proba([[employee_id, department, region, education, gender, recruitment_channel, no_of_trainings, age, previous_year_rating, length_of_service, awards_won, avg_training_score]])
        result_str += "NO PROMOTION " + "{:.4f}".format(output[0][0]) + "<br />"
        result_str += "PROMOTION " + "{:.4f}".format(output[0][1]) + "<br />"
    return result_str

@app.route('/input', methods=['GET'])
def input():
    return render_template('input.html')

@app.route('/model-api', methods=['POST'])
def model_api():
    employee_id = request.form.get('employee_id')
    department = request.form.get('department')
    region = request.form.get('region')
    education = request.form.get('education')
    gender = request.form.get('gender')
    recruitment_channel = request.form.get('recruitment_channel')
    no_of_trainings = request.form.get('no_of_trainings')
    age = request.form.get('age')
    previous_year_rating = request.form.get('previous_year_rating')
    length_of_service = request.form.get('length_of_service')
    awards_won = request.form.get('awards_won')
    avg_training_score = request.form.get('avg_training_score')

    employee_id	= int(employee_id)
    region = int(region)
    no_of_trainings = int(no_of_trainings)
    age = int(age)
    previous_year_rating = int(previous_year_rating)
    length_of_service = int(length_of_service)
    awards_won = int(awards_won)
    avg_training_score = int(avg_training_score)

    the_output = the_model(employee_id, department, region, education, gender, recruitment_channel, no_of_trainings, age, previous_year_rating, length_of_service, awards_won, avg_training_score)

    return the_output

if __name__ == '__main__':
    app.run(debug=True, port='8080', host='0.0.0.0', use_reloader=True)