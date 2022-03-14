import pandas as pd
import joblib

class Utils:
    def load_from_csv(self, path):
        return pd.read_csv(path)

    def load_from_mysql(self):
        pass

    def features_target(self, dataset, drop_cols, y_col):
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[y_col]

        return X, y

    def model_export(self, model, score):
        print(score)
        joblib.dump(model, './models/best_model.pkl')
