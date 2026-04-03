class Pipeline:
    def __init__(self, model, trainer, evaluator):
        self.model = model
        self.trainer = trainer
        self.evaluator = evaluator
    
    def run(self, X_train, y_train, X_test, y_test):
        self.trainer.train(self.model, X_train, y_train)
        predictions = self.model.predict(X_test)
        evaluation_results = self.evaluator.evaluate(predictions, y_test)
        return evaluation_results