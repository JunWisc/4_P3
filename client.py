import grpc
import modelserver_pb2
import modelserver_pb2_grpc
import csv
import sys
import threading

# some of the codes are from Microsoft Bing Chat



class WorkerThread(threading.Thread):
    def __init__(self, filename, stub):
        threading.Thread.__init__(self)
        self.filename = filename
        self.stub = stub
        self.hits = 0
        self.misses = 0

    def run(self):
        with open(self.filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                X=[float(x) for x in row]
                response = self.stub.Predict(modelserver_pb2.PredictRequest(X=X))
                if response.hit:
                    self.hits += 1
                else:
                    self.misses += 1
                    print(self.misses)

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 client.py <server_address> <port> <coefs> <filenames>")
        return

    server_address_port = 'localhost:5440'
    coefs = [1.0,2.0,3.0]
    filenames = ['workload/workload1.csv', 'workload/workload2.csv']

    channel = grpc.insecure_channel(server_address_port)
    stub = modelserver_pb2_grpc.ModelServerStub(channel)

    stub.SetCoefs(modelserver_pb2.SetCoefsRequest(coefs=coefs))

    threads = []
    for filename in filenames:
        thread = WorkerThread(filename, stub)
        thread.start()
        threads.append(thread)

    total_hits = 0
    total_misses = 0
    for thread in threads:
        thread.join()
        total_hits += thread.hits
        total_misses += thread.misses

    hit_rate = total_hits / (total_hits + total_misses)
    print(hit_rate)

if __name__ == '__main__':
    main()
