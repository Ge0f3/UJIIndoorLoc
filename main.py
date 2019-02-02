import numpy as np 
from flask import jsonify,abort,request,Flask
import pickle,joblib


my_predict = pickle.load(open('model_file','rb'))
ser_countvect =pickle.load(open('countvect','rb'))

#using joblib
# my_predict = joblib.load('model.pkl')
# ser_countvect=joblib.load('countvect.pkl')

app=Flask(__name__)

@app.route('/api',methods=['post'])
def predict():
	word = request.get_json(force=True)
	x_count = ser_countvect.transform(word)
	res = my_predict.predict(x_count)
	output = res[0]
	return jsonify(results=output)

if __name__ == '__main__':
	app.run(port=5004,debug=True)