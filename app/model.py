import pickle
import os

import xgboost as xgb
import pandas as pd
import numpy as np


class Model:
    def __init__(self):
        self.reg = xgb.XGBRegressor()
        rel_path_model = 'app/best_model.json'
        rel_path_cols = 'app/final_columns.pkl'
        rel_path_feat_imp = 'app/feat_imp.pkl'
        self.model_path = os.path.join(os.getcwd(), rel_path_model)
        self.columns_path = os.path.join(os.getcwd(), rel_path_cols)
        self.feat_imp_path = os.path.join(os.getcwd(), rel_path_feat_imp)
        self.cols = None
        self.load_model()
        self.load_columns()
        self.save_feat_importance()

    def load_model(self):
        # print(os.getcwd())
        self.reg.load_model(self.model_path)
        # trying with the pickle model
        # with open(self.model_path,'rb') as f:
        #     self.reg = pickle.load(f)

    def load_columns(self):
        with open(self.columns_path, 'rb') as f:
            self.cols = pickle.load(f)

    def save_feat_importance(self):
        # save the top 50 features
        if not os.path.exists(self.feat_imp_path):
            imp_feat = self.reg.feature_importances_
            df = pd.DataFrame()
            df['feature'] = self.cols['final_columns']
            df['importance'] = imp_feat
            df.sort_values(['importance'],
                           axis=0, ascending=False, inplace=True)
            df = df.head(50)
            importance = {}
            for i in range(50):
                importance[df.iloc[i]['feature']] = df.iloc[i]['importance']
            with open(self.feat_imp_path, 'wb') as f:
                pickle.dump(importance, f)

    def predict(self, input_data, stats_path=None):
        input_data = np.array(input_data).reshape(1, -1)
        input_pd = pd.DataFrame(input_data, columns=self.cols['final_columns'])
        time_to_eruption = self.reg.predict(input_pd)
        time_to_eruption = time_to_eruption[0]
        if stats_path is not None:
            input_pd.to_csv(stats_path, index=False)
        return time_to_eruption
