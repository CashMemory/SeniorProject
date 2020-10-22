from flask import Flask
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)


class HelloWorld(Resource):
    def get(self):
        return {'hello':'world'}

api.add_resource(HelloWorld,'/')

# class SessionAPI(Resource):
#     def get(self, id):
#         pass

#     def put(self, id):
#         pass

#     def delete(self, id):
#         pass

# api.add_resource(SessionAPI, '/workout/<int:id>', endpoint = 'session')


class CurlAPI(Resource):
    def get(self,id):
        return {'curl':f'{id}'}
    def post(self):
        

    def put(self,id):
        return{'put':id}
        pass

    def delete(self):
        pass

api.add_resource(CurlAPI, '/curl')

if __name__ == '__main__':
    app.run()