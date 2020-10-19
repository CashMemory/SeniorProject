import json

class Workout():

    def __init__(self):
        self.time = 0
        self.heart_rate = 0
        self.exercises = []
        self.prev_ex = None
        self.next_ex = None
        self.category = ""

class WorkoutEncoder(json.JSONEncoder):
    def default(self,o):
        return o.__dict__
