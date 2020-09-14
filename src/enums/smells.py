from enum import Enum, unique

@unique
class Smells(Enum):

    """List of available smells in the system"""

    DATA_CLASS = "data class"
    FEATURE_ENVY = "feature envy"
    GOD_CLASS = "god class"
    LONG_METHOD = "long method"

    def description(self):
        if(self == Smells.DATA_CLASS):
            # https://refactoring.guru/smells/data-class
            return "A data class refers to a class that contains only fields and crude methods for accessing them (getters and setters)."
        if(self == Smells.FEATURE_ENVY):
            # https://refactoring.guru/smells/feature-envy
            return "A method accesses the data of another object more than its own data."
        if(self == Smells.GOD_CLASS):
            return "A class that knows too much or does too much. Also known as large class"
        if(self == Smells.LONG_METHOD):
            return "A method contains too many lines of code. Usually more than 10"

    def label(self):
        """ The label used for each smell in a list """
        return str(self.value)

    def getDatasetFileName(self):
        """ returns the filename of the dataset that belongs to a smell """
        return str(self.value).replace(" ", "-") + ".arff"

    @staticmethod
    def getList():
        return list(map(str, Smells))
