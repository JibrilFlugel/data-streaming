import numpy as np
from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.streaming.context import StreamingContext
from pyspark.streaming.dstream import DStream
from pyspark.ml.linalg import DenseVector
from transforms.transforms import Transforms
from trainer import SparkConfig
import json

class DataLoader:
    def __init__(self, 
                 sparkContext: SparkContext, 
                 sparkStreamingContext: StreamingContext, 
                 sqlContext: SQLContext,
                 sparkConf: SparkConfig, 
                 transforms: Transforms) -> None:
        
        self.sc = sparkContext
        self.ssc = sparkStreamingContext
        self.sparkConf = sparkConf
        self.sql_context = sqlContext
        self.stream = self.ssc.socketTextStream(
            hostname=self.sparkConf.stream_host, 
            port=self.sparkConf.port,
        )
        self.transforms = transforms

    def parse_stream(self) -> DStream:
        try:
            json_stream = self.stream.map(lambda line: json.loads(line))
            json_stream.foreachRDD(lambda rdd: print(f"Received JSON batch: {rdd.count()} items"))
            json_stream_exploded = json_stream.flatMap(lambda x: x.values())
            json_stream_exploded.foreachRDD(lambda rdd: print(f"Exploded batch: {rdd.count()} items"))
            json_stream_exploded = json_stream_exploded.map(lambda x: list(x.values()))
            pixels = json_stream_exploded.map(lambda x: [np.array(x[:-1]).reshape(28, 28).astype(np.uint8), x[-1]])
            pixels = DataLoader.preprocess(pixels, self.transforms)
            pixels.foreachRDD(lambda rdd: print(f"Processed batch: {rdd.count()} items"))
            return pixels
        except Exception as e:
            print(f"Error parsing stream: {e}")
            raise

    @staticmethod
    def preprocess(stream: DStream, transforms: Transforms) -> DStream:
        try:
            stream = stream.map(lambda x: [transforms.transform(x[0]).reshape(-1).tolist(), x[1]])
            stream = stream.map(lambda x: [DenseVector(x[0]), x[1]])
            return stream
        except Exception as e:
            print(f"Error preprocessing stream: {e}")
            raise