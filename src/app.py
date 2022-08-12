# import Flask and jsonify
from flask import Flask, request
import pandas as pd
# import Resource, Api and reqparser
from flask_restful import Resource, Api, reqparse
import pickle
app = Flask(__name__)
api = Api(app)
model = pickle.load(open(r"pipeline_final.pkl", "rb" ))
class prediction(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        res = model.predict(df)
        return res.tolist()
api.add_resource(prediction, '/prediction')
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
