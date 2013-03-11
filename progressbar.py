# from
# http://stackoverflow.com/questions/5299950/separating-progress-tracking-and-loop-logic

from sys import stdout

# Progress meter
class ProgressMeter:
    def __init__(self, total):
        self.total = total
    def update(self, pc):
        decade = pc*10/self.total
        status_string = "\r["+"="*decade+" "*(9-decade)+"] %d of %d"%(pc,self.total)
        stdout.write(status_string)
        stdout.flush()

# Progress bar class
class RealProgressBar:
    def __init__(self):
        self.pm = None
        self.pc = 0
    def setMaximum(self, size):
        self.pm = ProgressMeter(total=size)
    def progress(self):
        self.pc += 1
        if self.pc % 100 == 0:
            self.pm.update(self.pc)

# a fake progress bar that does nothing
class NoProgressBar:
    def setMaximum(self, size):
         pass 
    def progress(self):
         pass

# turn a collection into one that shows progress when iterated
def withProgress(collection, progressBar=NoProgressBar()):
      progressBar.setMaximum(len(collection))
      for element in collection:
           progressBar.progress();
           yield element
