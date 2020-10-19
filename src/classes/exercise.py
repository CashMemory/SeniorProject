import json

class Exercise():

    def __init__(self):
        self.name = ""
        self.ID = 0
        self.form = None
        self.joints = []
        self.reps = 0
        self.RepCounter = None
        self.RepStack = []

class ExerciseEncoder(json.JSONEncoder):
    def default(self,o):
        return o.__dict__
