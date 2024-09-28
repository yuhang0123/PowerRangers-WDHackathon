import joblib

class Game:
    def __init__(self, name, cpu, ram, battery, latency):
        self.name = name
        self.cpu = cpu
        self.ram = ram
        self.battery = battery
        self.latency = latency

    def handler(self):
        return [self.name, self.cpu, self.ram, self.battery, self.latency]
    
class Application:
    def __init__(self):
        self.games_running = {}
        self.model = joblib.load('my_model.pkl')

    def create_games_object(self, name, cpu, ram , battery, latency):
        new_game = Game(name=name, cpu=cpu, ram=ram, battery=battery, latency=latency)
        new_game_data = new_game.handler()
        pred_data = self.model.predict(new_game_data)
        return pred_data


    def calculate_GPU_usage(self):
        pass

    def display(self):
        tab = "  "
        ret_val = "  Name  |  CPU_USAGE  | RAM_USAGE  |  BATTERY  |  LATENCY  |  GPU_USAGE  \n"
        for game, gpu_usage in self.games_running:
            ret_val += tab + game.name + tab + "|" +\
                       tab + str(game.cpu) + tab + "|" +\
                       tab + str(game.ram) + tab + "|" +\
                       tab + str(game.battery) + tab + "|" +\
                       tab + str(game.latency) + tab + "|" +\
                       tab + str(gpu_usage) + tab + "\n"
            
        print(ret_val)

if __name__ == '__main__':
    app = Application()
    pred = app.create_games_object("A", 5.121085, 99.646115, 0.937204, 151.423450)
    print(pred)
