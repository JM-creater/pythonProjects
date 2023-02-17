# pip install flask
# pip install flask-cors
from flask import Flask, request, Response, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

def the_model(breed, height_inches, weight_lbs, has_testicles, has_vagina, leg_raise, squatting_down):
	result_str = ''
	with open('logistic.pk', 'rb') as f:
		model = pickle.load(f)

		if breed == 'Anatolian Sheepdog':
			breed = 1
		elif breed == 'Bernese Mountain Dog':
			breed = 2
		elif breed == 'Bloodhound':
			breed = 3
		elif breed == 'Borzoi':
			breed = 4
		elif breed == 'Bullmastiff':
			breed = 5
		elif breed == 'Great Dane':
			breed = 6
		elif breed == 'Great Pyrenees':
			breed = 7
		elif breed == 'Great Swiss Mountain Dog':
			breed = 8
		elif breed == 'Irish Wolfhound':
			breed = 9
		elif breed == 'Kuvasz':
			breed = 10
		elif breed == 'Mastiff':
			breed = 11
		elif breed == 'Neopolitan Mastiff':
			breed = 12
		elif breed == 'Newfoundland':
			breed = 13
		elif breed == 'Otter Hound':
			breed = 14
		elif breed == 'Rottweiler':
			breed = 15
		elif breed == 'Saint Bernard':
			breed = 16
		elif breed == 'Afghan Hound':
			breed = 17
		elif breed == 'American Foxhound':
			breed = 18
		elif breed == 'Beauceron':
			breed = 19
		elif breed == 'Belgian Malinois':
			breed = 20
		elif breed == 'Belgian Sheepdog':
			breed = 21
		elif breed == 'Belgian Tervuren':
			breed = 22
		elif breed == 'Black And Tan Coonhound':
			breed = 23
		elif breed == 'Black And Tan Coonhound':
			breed = 24
		elif breed == 'Black Russian Terrier':
			breed = 25
		elif breed == 'Bouvier Des Flandres':
			breed = 26
		elif breed == 'Boxer':
			breed = 27
		elif breed == 'Briard':
			breed = 28
		elif breed == 'Chesapeake Bay Retriever':
			breed = 29
		elif breed == 'Clumber Spaniel':
			breed = 30
		elif breed == 'Collie (Rough) & (Smooth)':
			breed = 31
		elif breed == 'Curly Coated Retriever':
			breed = 32
		elif breed == 'Doberman Pinscher':
			breed = 33
		elif breed == 'English Foxhound':
			breed = 34
		elif breed == 'English Setter':
			breed = 35
		elif breed == 'German Shepherd Dog':
			breed = 36
		elif breed == 'German Shorthaired Pointer':
			breed = 37
		elif breed == 'German Wirehaired Pointer':
			breed = 38
		elif breed == 'Giant Schnauzer':
			breed = 39
		elif breed == 'Golden Retriever':
			breed = 40
		elif breed == 'Gordon Setter':
			breed = 41
		elif breed == 'Greyhound':
			breed = 42
		elif breed == 'Irish Setter':
			breed = 43
		elif breed == 'Komondor':
			breed = 44
		elif breed == 'Labrador Retriever':
			breed = 45
		elif breed == 'Old English Sheepdog (Bobtail)':
			breed = 46
		elif breed == 'Rhodesian Ridgeback':
			breed = 47
		elif breed == 'Scottish Deerhound':
			breed = 48
		elif breed == 'Spinone Italiano':
			breed = 49
		elif breed == 'Tibetan Mastiff':
			breed = 50
		elif breed == 'Poodle Standard':
			breed = 51
		elif breed == 'Weimaraner':
			breed = 52
		elif breed == 'Airdale Terrier':
			breed = 53
		elif breed == 'American Staffordshire Terrier':
			breed = 54
		elif breed == 'American Water Spaniel':
			breed = 55
		elif breed == 'Australian Cattle Dog':
			breed = 56
		elif breed == 'Australian Shepherd':
			breed = 57
		elif breed == 'Basset Hound':
			breed = 58
		elif breed == 'Bearded Collie':
			breed = 59
		elif breed == 'Border Collie':
			breed = 60
		elif breed == 'Brittany':
			breed = 61
		elif breed == 'Bull Dog':
			breed = 62
		elif breed == 'Bull Terrier':
			breed = 63
		elif breed == 'Canaan Dog':
			breed = 64
		elif breed == 'Chinese Shar Pei':
			breed = 65
		elif breed == 'Chow Chow':
			breed = 66
		elif breed == 'Cocker Spaniel-American':
			breed = 67
		elif breed == 'Cocker Spaniel-English':
			breed = 68
		elif breed == 'Dalmatian':
			breed = 69
		elif breed == 'English Springer Spaniel':
			breed = 70
		elif breed == 'Field Spaniel':
			breed = 71
		elif breed == 'Flat Coated Retriever':
			breed = 72
		elif breed == 'Finnish Spitz':
			breed = 73
		elif breed == 'Harrier':
			breed = 74
		elif breed == 'Ibizan Hound':
			breed = 75
		elif breed == 'Irish Terrier':
			breed = 76
		elif breed == 'Irish Water Spaniel':
			breed = 77
		elif breed == 'Keeshond':
			breed = 78
		elif breed == 'Kerry Blue Terrier':
			breed = 79
		elif breed == 'Norwegian Elkhound':
			breed = 80
		elif breed == 'Nova Scotia Duck Tolling Retriever':
			breed = 81
		elif breed == 'Petit Basset Griffon Vendeen':
			breed = 82
		elif breed == 'Pharaoh Hound':
			breed = 83
		elif breed == 'Plott Hound':
			breed = 84
		elif breed == 'Pointer':
			breed = 85
		elif breed == 'Polish Lowland Sheepdog':
			breed = 86
		elif breed == 'Portuguese Water Dog':
			breed = 87
		elif breed == 'Redbone Coonhound':
			breed = 88
		elif breed == 'Saluki':
			breed = 89
		elif breed == 'Samoyed':
			breed = 90
		elif breed == 'Siberian Husky':
			breed = 91
		elif breed == 'Soft-Coated Wheaten Terrier':
			breed = 92
		elif breed == 'Staffordshire Bull Terrier':
			breed = 93
		elif breed == 'Standard Schnauzer':
			breed = 94
		elif breed == 'Sussex Spaniel':
			breed = 95
		elif breed == 'Vizsla':
			breed = 96
		elif breed == 'Welsh Springer Spaniel':
			breed = 97
		elif breed == 'Wirehaired Pointing Griffon':
			breed = 98
		elif breed == 'American Eskimo':
			breed = 99
		elif breed == 'Australian Terrier':
			breed = 100
		elif breed == 'Basenji':
			breed = 101
		elif breed == 'Beagle':
			breed = 102
		elif breed == 'Bedlington Terrier':
			breed = 103
		elif breed == 'Bichon Frise':
			breed = 104
		elif breed == 'Border Terrier':
			breed = 105
		elif breed == 'Boston Terrier':
			breed = 106
		elif breed == 'Brussels Griffon':
			breed = 107
		elif breed == 'Cairn Terrier':
			breed = 108
		elif breed == 'Cardigan Welsh Corgi':
			breed = 109
		elif breed == 'Cavalier King Charles Spaniel':
			breed = 110
		elif breed == 'Dachshund':
			breed = 111
		elif breed == 'Dandie Dinmont Terrier':
			breed = 112
		elif breed == 'English Toy Spaniel':
			breed = 113
		elif breed == 'Fox Terrier ?? Smooth':
			breed = 114
		elif breed == 'Fox Terrier ?? Wirehair':
			breed = 115
		elif breed == 'French Bulldog':
			breed = 116
		elif breed == 'German Pinscher':
			breed = 117
		elif breed == 'Glen Imaal Terrier':
			breed = 118
		elif breed == 'Lakeland Terrier':
			breed = 119
		elif breed == 'Manchester Terrier (Standard)':
			breed = 120
		elif breed == 'Poodle Miniature':
			breed = 121
		elif breed == 'Pug':
			breed = 122
		elif breed == 'Puli':
			breed = 123
		elif breed == 'Scottish Terrier':
			breed = 124
		elif breed == 'Sealyham Terrier':
			breed = 125
		elif breed == 'Shetland Sheepdog (Sheltie)':
			breed = 126
		elif breed == 'Shiba Inu':
			breed = 127
		elif breed == 'Shih Tzu':
			breed = 128
		elif breed == 'Chihuahua':
			breed = 129
		elif breed == 'Maltese':
			breed = 130
		elif breed == 'Pomeranian':
			breed = 131
		elif breed == 'Yorkshire Terrier':
			breed = 132
		else:
			breed = 0

		has_testicles = 1 if has_testicles == 'yes' else 0
		has_vagina = 1 if has_vagina == 'yes' else 0
		leg_raise = 1 if leg_raise == 'yes' else 0
		squatting_down = 1 if squatting_down == 'yes' else 0

		output = model.predict_proba([[breed, height_inches, weight_lbs, has_testicles, has_vagina, leg_raise, squatting_down]])
		result_str += "male " + "{:.4f}".format(output[0][0]) + "<br />"
		result_str += "female " + "{:.4f}".format(output[0][1]) + "<br />"
	return result_str

@app.route('/input', methods=['GET'])
def input():
	return render_template('input.html')

@app.route('/model-api', methods=['POST'])
def model_api():
		breed = request.form.get('breed')
		height_inches = request.form.get('height_inches')
		weight_lbs = request.form.get('weight_lbs')
		has_testicles = request.form.get('has_testicles')
		has_vagina = request.form.get('has_vagina')
		leg_raise = request.form.get('leg_raise')
		squatting_down = request.form.get('squatting_down')

		height_inches = int(height_inches)
		weight_lbs = int(weight_lbs)

		the_output = the_model(breed, height_inches, weight_lbs, has_testicles, has_vagina, leg_raise, squatting_down)

		return the_output

if __name__ == '__main__':
	app.run(debug=True, port='8080', host='0.0.0.0', use_reloader=True)
