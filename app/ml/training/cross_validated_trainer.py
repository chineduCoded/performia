from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

from ..split.student_splitter import StudentSplitter
from ..models.base import BaseModelSpec

class CrossValidatedTrainer:
    def __init__(self, model: BaseModelSpec, splitter: StudentSplitter):
        self.pipeline = model.build_pipeline()
        self.splitter = splitter

    def evaluate(self, X, y, student_ids):
        aucs, aps = [], []

        for train_idx, val_idx in self.splitter.split(X, y, student_ids):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            self.pipeline.fit(X_train, y_train)
            y_prob = self.pipeline.predict_proba(X_val)[:, 1]

            if len(np.unique(y_val)) > 1:
                aucs.append(roc_auc_score(y_val, y_prob))
                aps.append(average_precision_score(y_val, y_prob))

        return {
            "ROC_AUC": np.mean(aucs),
            "AP": np.mean(aps),
        }
