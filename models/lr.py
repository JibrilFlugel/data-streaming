from typing import List
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np

class LR:
    def __init__(self, penalty='l2', C=1.0, max_iter=100):
        if C <= 0:
            raise ValueError("C must be positive")
        self.penalty = penalty
        self.regParam = 1.0 / C
        self.max_iter = max_iter
        self.model = None

    def initialize_model(self):
        if self.model is None:
            try:
                self.model = LogisticRegression(
                    maxIter=self.max_iter,
                    regParam=self.regParam,
                    elasticNetParam=0.0 if self.penalty == 'l2' else 1.0,
                    labelCol="label",
                    featuresCol="image"
                )
            except Exception as e:
                print(f"Error initializing LogisticRegression: {e}")
                raise

    def train(self, df: DataFrame) -> List:
        if df.count() == 0:
            print("Error: Empty DataFrame received")
            return [], 0.0, 0.0, 0.0, 0.0

        self.initialize_model()
        try:
            fitted_model = self.model.fit(df)
        except Exception as e:
            print(f"Error fitting model: {e}")
            return [], 0.0, 0.0, 0.0, 0.0
        
        predictions_df = fitted_model.transform(df)
        predictions = predictions_df.select("prediction").rdd.flatMap(lambda x: x).collect()
        
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
        accuracy = evaluator.evaluate(predictions_df, {evaluator.metricName: "accuracy"})
        precision = evaluator.evaluate(predictions_df, {evaluator.metricName: "weightedPrecision"})
        recall = evaluator.evaluate(predictions_df, {evaluator.metricName: "weightedRecall"})
        f1 = evaluator.evaluate(predictions_df, {evaluator.metricName: "f1"})
        
        return predictions, accuracy, precision, recall, f1

    def predict(self, df: DataFrame) -> List:
        if df.count() == 0:
            print("Error: Empty DataFrame received")
            return [], 0.0, 0.0, 0.0, 0.0, np.zeros((10, 10))

        self.initialize_model()
        try:
            predictions_df = self.model.transform(df)
        except Exception as e:
            print(f"Error predicting: {e}")
            return [], 0.0, 0.0, 0.0, 0.0, np.zeros((10, 10))
        
        predictions = predictions_df.select("prediction").rdd.flatMap(lambda x: x).collect()
        
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
        accuracy = evaluator.evaluate(predictions_df, {evaluator.metricName: "accuracy"})
        precision = evaluator.evaluate(predictions_df, {evaluator.metricName: "weightedPrecision"})
        recall = evaluator.evaluate(predictions_df, {evaluator.metricName: "weightedRecall"})
        f1 = evaluator.evaluate(predictions_df, {evaluator.metricName: "f1"})
        
        cm = np.zeros((10, 10))
        prediction_and_labels = predictions_df.select("prediction", "label").rdd.map(lambda x: (x[0], x[1])).collect()
        for pred, label in prediction_and_labels:
            cm[int(label), int(pred)] += 1

        return predictions, accuracy, precision, recall, f1, cm