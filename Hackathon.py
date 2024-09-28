# import joblib and load the trained model
import joblib

class Application:
    def __init__(self) -> None:
        pass

    def load_model(self):
        self.model = joblib.load('my_model.pkl')

    def usage_prediction(self, games_data):

        result = {}

        for game in games_data:
            game_name = game[0]
            game_data = game[1:]
            pred_score = self.model.predict(game_data)
            result[game_name] = pred_score

        return result




class Game: 
    def __init__(self, name, CPU_usage, RAM_usage, battery_Usage, latency):
        self.name = name
        self.CPU_usage = CPU_usage
        self.RAM_usage = RAM_usage
        self.battery_usage = battery_Usage
        self.latency = latency

    def handler(self):
        return [self.name, self.CPU_usage, self.RAM_usage, self.battery_Usage, self.latency]

class Computer:
    def __init__(self, singleCorePerformance, multiCorePerformance, computerRAM):
        self.singleCorePerformance = singleCorePerformance
        self.multiCorePerformance = multiCorePerformance
        self.computerRAM = computerRAM

        
        




class displayApplication:
    def __init__(self) -> None:
        pass
    

