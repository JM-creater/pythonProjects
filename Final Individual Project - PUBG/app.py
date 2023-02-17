# pip install flask
# pip install flask-cors
from flask import Flask, request, Response, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

def the_model(wall_pen, HDMG, magazine, LDMG, price, BDMG):
	result_str = ''
	with open('logistic.pk', 'rb') as f:
		model = pickle.load(f)
		if wall_pen == 'Medium':
			wall_pen = 1
		elif wall_pen == 'High':
			wall_pen = 2
		else:
			wall_pen = 0

		output = model.predict_proba([[wall_pen, HDMG, magazine, LDMG, price, BDMG]])
		if output[0][0] > output[0][1] and output[0][0] > output[0][2] and output[0][0] > output[0][3] and output[0][0] > output[0][4] and output[0][0] > output[0][5]:
			result_str += "THE HIGHEST PERCENTAGE IS <span class='test'>PISTOL</span>" + "<br />" + "<br />"
		elif output[0][1] > output[0][0] and output[0][1] > output[0][2] and output[0][1] > output[0][3] and output[0][1] > output[0][4] and output[0][1] > output[0][5]:
			result_str += "THE HIGHEST PERCENTAGE IS <span class='test'>SHOTGUN</span> " + "<br />" + "<br />"
		elif output[0][2] > output[0][1] and output[0][2] > output[0][0] and output[0][2] > output[0][3] and output[0][2] > output[0][4] and output[0][2] > output[0][5]:
			result_str += "THE HIGHEST PERCENTAGE IS <span class='test'>SUBMACHINE GUN </span>" + "<br />" + "<br />"
		elif output[0][3] > output[0][1] and output[0][3] > output[0][0] and output[0][3] > output[0][2] and output[0][3] > output[0][4] and output[0][3] > output[0][5]:
			result_str += "THE HIGHEST PERCENTAGE IS <span class='test'>ASSAULT RIFLE </span>" + "<br />" + "<br />"
		elif output[0][4] > output[0][1] and output[0][4] > output[0][0] and output[0][4] > output[0][2] and output[0][4] > output[0][3] and output[0][4] > output[0][5]:
			result_str += "THE HIGHEST PERCENTAGE IS <span class='test'>SNIPER RIFLE </span>" + "<br />" + "<br />"
		elif output[0][5] > output[0][0] and output[0][5] > output[0][1] and output[0][5] > output[0][2] and output[0][5] > output[0][3] and output[0][5] > output[0][4]:
			result_str += "THE HIGHEST PERCENTAGE IS <span class='test'>HEAVY </span>" + "<br />" + "<br />"

		result_str += "PISTOL " + "<span class='percent'>{:.4f}</span>".format(output[0][0]*100) + "<span class='percent'>%</span>" + "<br />"
		result_str += "SHOTGUN " + "<span class='percent'>{:.4f}</span>".format(output[0][1]*100) + "<span class='percent'>%</span>" + "<br />"
		result_str += "SUBMACHINE GUN " + "<span class='percent'>{:.4f}</span>".format(output[0][2]*100) + "<span class='percent'>%</span>" + "<br />"
		result_str += "ASSAULT RIFLE " + "<span class='percent'>{:.4f}</span>".format(output[0][3]*100) + "<span class='percent'>%</span>" + "<br />"
		result_str += "SNIPER RIFLE " + "<span class='percent'>{:.4f}</span>".format(output[0][4]*100) + "<span class='percent'>%</span>" + "<br />"
		result_str += "HEAVY " + "<span class='percent'>{:.4f}</span>".format(output[0][5]*100) + "<span class='percent'>%</span>" + "<br />"
	return result_str

@app.route('/sample-url', methods=['GET'])
def sample_url():
	return render_template('sample-url.html')

@app.route('/input', methods=['GET'])
def input():
	return render_template('input.html')

@app.route('/model-api', methods=['POST'])
def model_api():
	wall_pen = request.form.get('wall_pen')
	HDMG = request.form.get('HDMG')
	magazine = request.form.get('magazine')
	LDMG = request.form.get('LDMG')
	price = request.form.get('price')
	BDMG = request.form.get('BDMG')

	HDMG = float(HDMG)
	magazine = int(magazine)
	LDMG = float(LDMG)
	price = int(price)
	BDMG = float(BDMG)
	the_output = the_model(wall_pen, HDMG, magazine, LDMG, price, BDMG)
	return the_output

if __name__ == '__main__':
	app.run(debug=True, port='8080', host='0.0.0.0', use_reloader=True)