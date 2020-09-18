import os
from sklearn.impute import SimpleImputer
from src.enums.smells import SystemSmells
from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

class DataRepo():

    # Constructor
    def __init__(self, *args, **kwargs):

        """ CLASS CONSTRUCTOR and class attributes"""

        self.smell = None
        self.data = None
        self.numberOfRows = 0
        self.percentageTesting = 0.15
        self.trainingData = {
            "x": None,
            "y": None,
        }
        self.testingData = {
            "x": None,
            "y": None,
        }

        if 'percentageTesting' in kwargs:
            percentage = kwargs['percentageTesting']
            if percentage > 0 and percentage < 100:
                # user can enter values between 0 - 100
                self.percentageTesting = percentage / 100
            else:
                # todo throw error
                pass

        if 'smell' in kwargs and kwargs['smell'] in SystemSmells:
            self.smell = kwargs['smell']

        if 'percentageTesting' in kwargs:
            self.percentageTesting = kwargs['percentageTesting']

    def getDatasetPath(self):

        """Class in charge of getting or loading datasets, cleaning them and then delivering to the classes training the models """

        if self.smell is not None:
            cwd = os.getcwd()
            fileName = self.smell.getDatasetFileName()
            return os.path.join(cwd,"data", fileName);

        return None

    def loadDataset(self, *args, **kwargs):

        path = self.getDatasetPath()

        if path is None:
            #todo throw error
            pass
        else:
            #1. load data
            self.data = arff.loadarff(path)
            df = pd.DataFrame(self.data[0])
            index = df.index
            self.numberOfRows = len(index)

            #2. get data for X and clean it if data is missing
            if len(df.isnull().sum().index) > 0:
                xData = self.cleanDataX(df)
            else:
                xData = df.iloc[:, :-1].values

            #3. get data for Y
            yOriginalData = df.iloc[:, -1].values

            # transform Y dependent data to binary equivalent
            yData = MultiLabelBinarizer().fit_transform(yOriginalData)

            if 'convertYToInt' in kwargs and kwargs['convertYToInt']:
                labelEncoder = preprocessing.LabelEncoder()
                yDataB = labelEncoder.fit_transform(yOriginalData)
                self.setTestingAndTrainingData(xData, yDataB)
            else:
                #4. generate trainig and test data
                self.setTestingAndTrainingData(xData, yData)

    def setTestingAndTrainingData(self, xData, yData):

        """ SETTER FOR TRAINING AND TESTING DATA """

        self.trainingData['x'], self.testingData['x'], self.trainingData['y'],  self.testingData['y'] = train_test_split(xData, yData, test_size=self.percentageTesting, random_state=42)

    def cleanDataX(self, df):
        """ Cleans the data in X and fixes missing data by using the median approach """
        # TODO. find condition to avoid always having to do this
        xDataCopy = df.iloc[:, :-1].copy()
        imputer = SimpleImputer(strategy="median")
        imputer.fit(xDataCopy)
        xData = imputer.transform(xDataCopy)
        # new_X_df = pd.DataFrame(xData, columns=xDataCopy.columns, index=xDataCopy.index)
        # new_X_df.info()
        return xData

