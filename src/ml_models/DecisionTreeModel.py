from src.ml_models.mlModel import MLModel

class DecisionTree(MLModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def trainModel(self, *args, **kwargs):

        print("intro train models decision tree")
