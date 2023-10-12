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
            for row in reader:
                X=[float(x) for x in row]
                response = self.stub.Predict(modelserver_pb2.PredictRequest(X=X))
                if response is not None:
                    if response.hit:
                        with self.lock:
                            self.hits += 1
                    else:
                        with self.lock:
                            self.misses +=1

def main():

    port="localhost:"+str(sys.argv[1])
    coefs_prev=sys.argv[2].split(",")
    coefs=[]
    for coef in coefs_prev:
        coefs.append(float(coef))
    filenames = sys.argv[3:]

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

