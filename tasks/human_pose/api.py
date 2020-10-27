
from flask import Flask

from flask_restful import Api, Resource

from get_video import LeftBicepCurl, executing

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



