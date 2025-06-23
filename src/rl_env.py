# Standard Libraries
import numpy as np
import pandas as pd

# Scikit-learn tools
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

class CreditLimitEnv:
    def __init__(self, df):
        # filter and reset data
        self.df = df.reset_index(drop=True)
        self.df = self.df[self.df['BALANCE_CLASS'].isin([0, 1])].reset_index(drop=True)
        self.current_step = 0
        self.state_space = ['BALANCE_CLASS', 'UR', 'PR', 'D_PROVISION_bin']
        self.action_space = [0, 1]
        self.n_customers = len(self.df)

        # params
        self.beta = 0.2
        self.pd_dict = {0: 0.01, 1: 0.05, 2: 0.15}
        self.lgd = 0.5
        self.ccf = 0.8

        global provision_bins

        # train bal_3 predictors
        self.train_bal_3_predictors()

    def train_bal_3_predictors(self):
        global ccb, provision_bins

        # data preparation
        prospective = ccb[ccb['MONTHS_BALANCE'].isin([-1, -2, -3])]
        bal_3 = prospective.groupby('SK_ID_CURR')['AMT_BALANCE'].mean().reset_index().rename(
            columns={'AMT_BALANCE': 'BAL_3'})
        train_df = self.df.merge(bal_3, on='SK_ID_CURR', how='left').fillna(0)

        # categorize BALANCE_CLASS
        train_0 = train_df[train_df['BALANCE_CLASS'] == 0]
        train_1 = train_df[train_df['BALANCE_CLASS'] == 1]

        # remove L_R
        predictor_features = self.state_space + ['UR', 'PR', 'INT', 'L_P']  # 移除了L_R

        # grid search params
        param_grid = {
            'n_estimators': [100, 200, 300],  #
            'max_depth': [10, 15, 20, None],  #
            'min_samples_leaf': [1, 5, 10],  #
            'max_features': ['auto', 'sqrt', 0.8]  #
        }

        # train 0 model
        if not train_0.empty:
            X_0, y_0 = train_0[predictor_features], train_0['BAL_3']

            # grid search
            grid_search_0 = GridSearchCV(
                estimator=RandomForestRegressor(random_state=42),
                param_grid=param_grid,
                cv=3,
                scoring='r2',
                n_jobs=-1
            )
            grid_search_0.fit(X_0, y_0)
            self.regressor_0 = grid_search_0.best_estimator_

            print(f"Class 0 best_params: {grid_search_0.best_params_}")
            print(f"Class 0 test R²: {grid_search_0.best_score_:.4f}")
            print(f"Class 0 train R²: {self.regressor_0.score(X_0, y_0):.4f}")
        else:
            self.regressor_0 = None
            print("Class 0无训练数据")

        # train 1 model
        if not train_1.empty:
            X_1, y_1 = train_1[predictor_features], train_1['BAL_3']

            grid_search_1 = GridSearchCV(
                estimator=RandomForestRegressor(random_state=42),
                param_grid=param_grid,
                cv=3,
                scoring='r2',
                n_jobs=-1
            )
            grid_search_1.fit(X_1, y_1)
            self.regressor_1 = grid_search_1.best_estimator_

            print(f"Class 1 best_params: {grid_search_1.best_params_}")
            print(f"Class 1 test R²: {grid_search_1.best_score_:.4f}")
            print(f"Class 1 train R²: {self.regressor_1.score(X_1, y_1):.4f}")
        else:
            self.regressor_1 = None
            print("Class 1无训练数据")

    def reset(self):
        self.current_step = np.random.randint(0, self.n_customers)
        state = self.df.iloc[self.current_step][self.state_space].values
        self.current_l_p = self.df.iloc[self.current_step]['L_P']
        self.current_bal = self.df.iloc[self.current_step]['AVG_BALANCE']
        self.current_int = self.df.iloc[self.current_step]['INT']
        return state

    def step(self, action):
        global provision_bins

        # update credit limit
        new_l_p = self.current_l_p * (1 + self.beta) if action == 1 else self.current_l_p

        # get current status
        balance_class = self.df.iloc[self.current_step]['BALANCE_CLASS']
        predictor_features = self.state_space + ['UR', 'PR', 'INT', 'L_P']  # 移除了L_R
        state_data_row = self.df.iloc[self.current_step][predictor_features].values.reshape(1, -1)

        # predict bal_3
        bal_3_pred = 0
        if balance_class == 0 and self.regressor_0:
            bal_3_pred = max(0, self.regressor_0.predict(state_data_row)[0])
        elif balance_class == 1 and self.regressor_1:
            bal_3_pred = max(0, self.regressor_1.predict(state_data_row)[0])

        pd_value = self.pd_dict[balance_class]

        # calculate reward
        reward = (
                3 * self.current_int * self.current_bal * (1 - pd_value)
                - pd_value * self.lgd * (bal_3_pred + self.ccf * (new_l_p - bal_3_pred))
        )
        reward = np.clip(reward, -1e6, 1e6) / 1e6

        # calculate new D_PROVISION_bin
        l_r_value = self.df.iloc[self.current_step]['L_R']
        new_delta_provision = (new_l_p - l_r_value) / l_r_value if l_r_value != 0 else 0

        new_dp_bin_result = pd.cut([new_delta_provision], bins=provision_bins, labels=False, include_lowest=True)
        new_dp_bin = int(new_dp_bin_result[0]) if new_dp_bin_result.size > 0 and not pd.isna(
            new_dp_bin_result[0]) else 0

        # update new status
        new_state = self.df.iloc[self.current_step][self.state_space].values.copy()
        dp_bin_index = self.state_space.index('D_PROVISION_bin')
        new_state[dp_bin_index] = new_dp_bin

        # update new env status
        self.current_step = (self.current_step + 1) % self.n_customers
        self.current_l_p = self.df.iloc[self.current_step]['L_P']
        self.current_bal = self.df.iloc[self.current_step]['AVG_BALANCE']
        self.current_int = self.df.iloc[self.current_step]['INT']

        done = self.current_step == 0
        return new_state, reward, done, {}

