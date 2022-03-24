class Logger:
    def __init__(self, pref = ">>"):
        self.pref = pref

    def error(self, msg):
        print("ERROR [{}]: {}".format(self.pref, msg))
        
    def log(self, msg):
        print("LOG [{}]: {}".format(self.pref, msg))