import pickle 
import os
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import seaborn as sns

class IntrusionDetector:
    def __init__(self, model_path=None, n_estimators=15):
        self.model = RandomForestClassifier(n_estimators=n_estimators)
        if model_path is not None:
            assert os.path.exists(model_path), "Model path not valid!!"
            with open(model_path, "rb") as model_file:
                self.model = pickle.load(model_file)
        self.classes = {
            0: "attack",
            1: "normal"
        }
        print("Model Loaded!")

    def predict(self, x, get_class_names=False):
        pred = self.model.predict(x)
        if get_class_names:
            return [self.classes[i] for i in pred]
        else:
            return pred
    
    def train(self, x, y):
        self.model.fit(x, y)

    def plot_metrics(self, y_true, y_pred):
        sns_plot = sns.heatmap(confusion_matrix(y_true, y_pred), annot=True)
        fig = sns_plot.get_figure()
        fig.savefig("confusion-matrix.png")
        print(f"Accuracy = {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision = {precision_score(y_true, y_pred):.4f}")
        print(f"Recall = {recall_score(y_true, y_pred):.4f}")
        print(f"ROC-AUC = {roc_auc_score(y_true, y_pred):.4f}")