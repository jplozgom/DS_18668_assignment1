from enum import Enum, unique

@unique
class SystemModels(Enum):

    """List of available models in the system"""

    DECISION_TREE = "Decision Tree"
    RANDOM_FOREST = "Random Forest"
    NAIVE_BAYES = "Naive Bayes"
    SVC_LINEAR = "SVC - Linear"
    SVC_POLYNOMIAL = "SVC - Polynomial"
    SVC_RBF = "SVC - RBF"
    SVC_SIGMOID = "SVC - Sigmoid"


    def description(self):
        if(self == SystemModels.DECISION_TREE):
            return "description of DECISION_TREE"
        elif(self == SystemModels.RANDOM_FOREST):
            return "description of RANDOM_FOREST"
        elif(self == SystemModels.NAIVE_BAYES):
            return "description of NAIVE_BAYES"
        elif(self == SystemModels.SVC_LINEAR):
            return "description of SVC_LINEAR"
        elif(self == SystemModels.SVC_POLYNOMIAL):
            return "description of SVC_POLYNOMIAL"
        elif(self == SystemModels.SVC_RBF):
            return "description of SVC_RBF"
        elif(self == SystemModels.SVC_SIGMOID):
            return "description of SVC_SIGMOID"

    def label(self):
        """ The label used for each model in a list """
        return str(self.value)

    def getDatasetFileName(self):
        """ returns the filename of the dataset that belongs to a model """
        return str(self.value).replace(" ", "-") + ".arff"

    @staticmethod
    def getList():
        return list(map(str, SystemModels))
