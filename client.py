import grpc
import modelserver_pb2
import modelserver_pb2_grpc
import csv
import sys
import threading
import time

# some of the codes are from Microsoft Bing Chat

class WorkerThread(threading.Thread):
    def __init__(self, filename, stub, lock):
        threading.Thread.__init__(self)
        self.filename = filename
        self.stub = stub
        self.lock = lock
        self.hits = 0
        self.misses = 0

    def run(self):
        with open(self.filename, 'r') as f:
            reader = csv.reader(f)
            #X=[]
            for row in reader:
            #    for x in  row:
             #       x=x.strip(' ')
              #      x=x.strip('-')
               #     X.append(float(x))
                X=[float(x) for x in row]
                
                response = self.stub.Predict(modelserver_pb2.PredictRequest(X=X))
                if response is not None and response.hit:
                    with self.lock:
                        self.hits += 1
                else:
                    with self.lock:
                        self.misses += 1

def main():

    port = 'localhost:5440'
    coefs = [1.0,2.0,3.0]
    filenames = ['workload/workload1.csv', 'workload/workload2.csv']

    channel = grpc.insecure_channel(port)
    stub = modelserver_pb2_grpc.ModelServerStub(channel)

    stub.SetCoefs(modelserver_pb2.SetCoefsRequest(coefs=coefs))

    lock = threading.Lock()

    threads = []
    for filename in filenames:
        thread = WorkerThread(filename, stub, lock)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    total_hits = 0
    total_misses = 0
    with lock:
        for thread in threads:
            total_hits += thread.hits
            total_misses += thread.misses

    hit_rate = total_hits / (total_hits + total_misses)
    print(hit_rate)

if __name__ == '__main__':
    main()

