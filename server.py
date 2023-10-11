import torch
from cachetools import LRUCache
from threading import Lock
import modelserver_pb2
import modelserver_pb2_grpc
import grpc
from concurrent import futures

class PredictionCache:
    def __init__(self):
        self.coefs = None
        self.cache = LRUCache(maxsize=10)
        self.lock = Lock()
        self.cache_limit = 10

    def SetCoefs(self, coefs):
        with self.lock:
            self.coefs = coefs
            self.cache.clear()
    def Predict(self, X):
        X_rounded = torch.round(X * 10000) / 10000
        X_tuple = tuple(X_rounded.flatten().tolist())

        with self.lock:
            if X_tuple in self.cache:
                return self.cache[X_tuple], True

            if self.coefs is None:
                raise ValueError("Coefficients have not been set.")

            y = X @ self.coefs
            self.cache[X_tuple] = y

            return y, False

class ModelServer(modelserver_pb2_grpc.ModelServerServicer):
    def __init__(self):
        self.prediction_cache = PredictionCache()

    def SetCoefs(self, request, context):
        coefs = torch.tensor(request.coefs).view(-1, 1)
        self.prediction_cache.SetCoefs(coefs)
        return modelserver_pb2.SetCoefsResponse(error="")


    def Predict(self, request, context):
        X = torch.tensor(request.X).view(1, -1)
        y, hit = self.prediction_cache.Predict(X)
        return modelserver_pb2.PredictResponse(y=y.item(), hit=hit, error="")

def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=(('grpc.so_reuseport', 0),))
    modelserver_pb2_grpc.add_ModelServerServicer_to_server(ModelServer(), server)
    server.add_insecure_port("[::]:5440")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    main()
