
from flask import Flask

from flask_restful import Api, Resource

from exercise import LeftBicepCurl
from get_video import executing

class CurlAPI(Resource):
    def get(self):

        curl = LeftBicepCurl()
        executing = True

       

        return {'curl':f'{id}'}
    def post(self):
        pass

    def put(self,id):
        return{'put':id}
        

    def delete(self):
        pass



