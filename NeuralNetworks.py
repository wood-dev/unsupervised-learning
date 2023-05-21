import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from util import splitData, getFullFilePath
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.metrics import accuracy_score
from warnings import simplefilter
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from warnings import filterwarnings
filterwarnings('ignore')

PRESET_HIDDEN_LAYERS = [(20, 20), (20, 20)]
PRESET_MAX_ITER = [1000, 500]
PRESET_SOLVER = ['adam', 'sgd']
RANDOM_SEED = 100

class NeuralNetworks:

    def __init__(self, id, title, X_train, X_test, y_train, y_test):
        self.identifier = id
        self.title = title
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def analyzePerformance(self, hidden_layer_sizes=None, max_iter=None, solver=None):

        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        if hidden_layer_sizes is None:
            hidden_layer_sizes = PRESET_HIDDEN_LAYERS[self.identifier-1]

        if max_iter is None:
            max_iter = PRESET_MAX_ITER[self.identifier-1]

        if solver is None:
            solver = PRESET_SOLVER[self.identifier-1]

        mlp = MLPClassifier(activation='relu', momentum=0.9,
            hidden_layer_sizes=hidden_layer_sizes, learning_rate='constant',
            max_iter=max_iter, solver=solver, early_stopping=False, random_state=RANDOM_SEED)

        time_before_training = time()
        mlp.fit(X_train, y_train)
        time_after_training = time()
        predictions = mlp.predict(X_test)
        time_after_predict = time()

        print("NMI score: %.6f" % normalized_mutual_info_score(y_test, predictions))

        print('NN %d training time: %f seconds.' % (self.identifier-1, time_after_training - time_before_training))
        print('NN %d lookup time: %f seconds.' % (self.identifier-1, time_after_predict - time_after_training))

        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))


    def analyzeTrainSize(self):

        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        accuracy_train = []
        accuracy_test = []

        mlp = MLPClassifier(activation='relu', momentum=0.9,
            hidden_layer_sizes=PRESET_HIDDEN_LAYERS[self.identifier-1], learning_rate='constant',
            max_iter=PRESET_MAX_ITER[self.identifier-1], solver=PRESET_SOLVER[self.identifier-1],random_state=100)

        train_range = range(50, 100, 5)
        for train_size in train_range:

            mlp.fit(X_train, y_train)

            predict_train = mlp.predict(X_train)
            predict_test = mlp.predict(X_test)

            accuracy_train.append(accuracy_score(y_train, predict_train))
            accuracy_test.append(accuracy_score(y_test, predict_test))

        fig, ax = plt.subplots()

        ax.plot(train_range, accuracy_train, color="red", label="Training Accuracy")
        ax.plot(train_range, accuracy_test, color="green", label="Testing Accuracy")

        plt.title('Neural Networks: Training Size vs Accuracy on {title}'.format(title = self.title) )
        plt.xlabel('Training Size (%)')
        plt.ylabel("Accuracy")
        plt.legend()

        # stating max point
        ymax = max(accuracy_test)
        xpos = accuracy_test.index(ymax)
        xmax = train_range[xpos]
        ax.annotate('({xmax}, {ymax})'.format(xmax=xmax, ymax=round(ymax,4)), xy=(xmax, ymax), ha='center', va='bottom', color='green')

        filename = 'NN-{id}-TrainSize-vs-Accuracy.png'
        plt.savefig(getFullFilePath(filename.format(id = self.identifier)), bbox_inches='tight')
        plt.close()

    @ignore_warnings(category=ConvergenceWarning)
    def analyzeBestParameter(self):

        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        mlp = MLPClassifier(activation='relu', momentum=0.9, random_state=100)

        parameter_space = {
            'solver': ['adam', 'sgd'],
            'hidden_layer_sizes': [(5), (10), (5, 5), (10, 10), (15, 15), (20,20), (25,25), (5, 5, 5), (10, 10, 10), (15, 15, 15), (20,20,20), (25,25,25), (10, 10, 10, 10)],
            'learning_rate': ['constant','adaptive'],
            'max_iter': [1000],
        }

        clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)

        clf.fit(X_train, y_train)

        print('Best parameters found:\n', clf.best_params_)

        predictions = clf.predict(X_test)
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))







