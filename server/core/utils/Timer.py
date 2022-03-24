import time
from utils.Logger import Logger

class Timer:
    def __init__(self):
        self.st = None
        self.prev = None
        self.en = None
        self.diff = None
        self.logger = Logger(str(self.__class__))
    
    def start(self):
        self.st = time.time()
        self.prev = self.st
        
    def stop(self):
        self.en = time.time()
    
    def elapsed(self):
        if self.st != None and self.en != None:
            return self.en - self.st
        return None

    def lap(self):
        if self.prev != None:
            curr = time.time()
            diff = curr - self.prev
            self.logger.log("{:.3f}s".format(diff))
            self.prev = curr
            return
        self.logger.error("Something went wrong")

    def clear(self):
        self.st = None
        self.en = None
        self.diff = None