from trainer import SparkConfig, Trainer
from transforms.transforms import Transforms
from transforms.normalize import Normalize
from models.lr import LR
from pyspark.sql import SparkSession

transforms = Transforms([
    Normalize(
        mean=(0.1307,), 
        std=(0.3081,)
    )
])

if __name__ == "__main__":
    spark_config = SparkConfig()
    spark = SparkSession.builder \
        .appName(spark_config.appName) \
        .config("spark.driver.memory", spark_config.driver_memory) \
        .config("spark.executor.heartbeatInterval", "3600s") \
        .config("spark.network.timeout", "3601s") \
        .master(f"{spark_config.host}[{spark_config.receivers}]") \
        .getOrCreate()
    
    # Get SparkContext from SparkSession
    sc = spark.sparkContext
    
    model = LR(penalty='l2', C=1.0, max_iter=100)
    trainer = Trainer(model, split="train", spark_config=spark_config, transforms=transforms, spark_context=sc)
    trainer.train()
    
    spark.stop()