from sklearn.model_selection import GroupKFold

class StudentSplitter:
    def __init__(self, n_splits=5):
        self.cv = GroupKFold(n_splits=n_splits)

    def split(self, X, y, student_ids):
        return self.cv.split(X, y, groups=student_ids)
