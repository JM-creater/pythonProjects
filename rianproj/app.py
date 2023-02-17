# pip install flask
# pip install flask-cors
from flask import Flask, request, Response, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

def the_model(bullet_type, damage, magazine, range, fire_rate, bullet_speed):
	result_str = ''
	with open('logistic.pk', 'rb') as f:
		model = pickle.load(f)
		if bullet_type == '5.56':
			bullet_type = 1
		elif bullet_type == '9':
			bullet_type = 2
		elif bullet_type == '0.45':
			bullet_type = 3
		elif bullet_type == '12':
			bullet_type = 4
		elif bullet_type == '0.3':
			bullet_type = 5
		else:
			bullet_type = 0

		output = model.predict_proba([[bullet_type, damage, magazine, range, fire_rate, bullet_speed]])
		if output[0][0] > output[0][1] and output[0][0] > output[0][2] and output[0][0] > output[0][3] and output[0][0] > output[0][4]:
			result_str += "THE HIGHEST PERCENTAGE IS <span class='test'>ASSAULT RIFLE</span>" + "<br />" + "<br />"
		elif output[0][1] > output[0][0] and output[0][1] > output[0][2] and output[0][1] > output[0][3] and output[0][1] > output[0][4]:
			result_str += "THE HIGHEST PERCENTAGE IS <span class='test'>SUBMACHINE GUN</span> " + "<br />" + "<br />"
		elif output[0][2] > output[0][1] and output[0][2] > output[0][0] and output[0][2] > output[0][3] and output[0][2] > output[0][4]:
			result_str += "THE HIGHEST PERCENTAGE IS <span class='test'>SHOTGUN </span>" + "<br />" + "<br />"
		elif output[0][3] > output[0][1] and output[0][3] > output[0][0] and output[0][3] > output[0][2] and output[0][3] > output[0][4]:
			result_str += "THE HIGHEST PERCENTAGE IS <span class='test'>SNIPER RIFLE </span>" + "<br />" + "<br />"
		elif output[0][4] > output[0][1] and output[0][4] > output[0][0] and output[0][4] > output[0][2] and output[0][4] > output[0][3]:
			result_str += "THE HIGHEST PERCENTAGE IS <span class='test'>PISTOL </span>" + "<br />" + "<br />"

		result_str += "ASSAULT RIFLE " + "{:.4f}".format(output[0][0]) + "%" + "<br />"
		result_str += "SUBMACHINE GUN " + "{:.4f}".format(output[0][1]) + "%" + "<br />"
		result_str += "SHOTGUN " + "{:.4f}".format(output[0][2]) + "%" + "<br />"
		result_str += "SNIPER RIFLE " + "{:.4f}".format(output[0][3]) + "%" + "<br />"
		result_str += "PISTOL " + "{:.4f}".format(output[0][4]) + "%" + "<br />"
	return result_str

@app.route('/sample-url', methods=['GET'])
def sample_url():
	return render_template('sample-url.html')

@app.route('/input', methods=['GET'])
def input():
	return render_template('input.html')

@app.route('/model-api', methods=['POST'])
def model_api():
	bullet_type = request.form.get('bullet_type')
	damage = request.form.get('damage')
	magazine = request.form.get('magazine')
	range = request.form.get('range')
	fire_rate = request.form.get('fire_rate')
	bullet_speed = request.form.get('bullet_speed')

	damage = int(damage)
	magazine = int(magazine)
	range = int(range)
	fire_rate = float(fire_rate)
	bullet_speed = int(bullet_speed)
	the_output = the_model(bullet_type, damage, magazine, range, fire_rate, bullet_speed)
	return the_output

if __name__ == '__main__':
	app.run(debug=True, port='8080', host='0.0.0.0', use_reloader=True)