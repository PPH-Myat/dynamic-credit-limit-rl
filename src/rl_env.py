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

        # Create binned UR, PR from 456 raw features
        self.df['UR'] = pd.cut(self.df['UR_456'], bins=[-np.inf, 0.25, 0.5, 0.75, 1.0, np.inf], labels=False,
                               include_lowest=True)
        self.df['PR'] = pd.cut(self.df['PR_456'], bins=[-np.inf, 0.25, 0.5, 0.75, 1.0, np.inf], labels=False,
                               include_lowest=True)

        self.current_step = 0
        self.state_space = ['BALANCE_CLASS', 'UR', 'PR', 'D_PROVISION_bin']
        self.action_space = [0, 1]
        self.n_customers = len(self.df)

        # parameters
        self.beta = 0.2
        self.pd_dict = {0: 0.01, 1: 0.05, 2: 0.15}
        self.lgd = 0.5
        self.ccf = 0.8

        global provision_bins

        # train regressors (simulate future balance)
        self.train_bal_3_predictors()

    def train_bal_3_predictors(self):
        global ccb, provision_bins

        # === Label BAL_3 (future avg balance): from -6, -5, -4 ===
        bal_3 = (
            ccb[ccb['MONTHS_BALANCE'].isin([-6, -5, -4])]
            .groupby('SK_ID_CURR')['AMT_BALANCE']
            .mean().reset_index()
            .rename(columns={'AMT_BALANCE': 'BAL_3'})
        )

        # === Features: from -9, -8, -7 ===
        past = ccb[ccb['MONTHS_BALANCE'].isin([-9, -8, -7])]
        balance_past = (
            past.groupby('SK_ID_CURR')['AMT_BALANCE']
            .mean().reset_index()
            .rename(columns={'AMT_BALANCE': 'BALANCE_MEAN_PAST_789'})
        )

        # Merge: only those with full past and future available
        train_df = (
            self.df
            .merge(bal_3, on='SK_ID_CURR', how='inner')
            .merge(balance_past, on='SK_ID_CURR', how='left')
            .fillna(0)
        )

        predictor_features = ['BALANCE_CLASS', 'UR_789', 'PR_789', 'INT_789', 'L_P_789', 'BALANCE_MEAN_PAST_789']

        train_0 = train_df[train_df['BALANCE_CLASS'] == 0]
        train_1 = train_df[train_df['BALANCE_CLASS'] == 1]

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_leaf': [1, 5],
            'max_features': ['auto', 'sqrt', 0.8]
        }

        if not train_0.empty:
            X_0, y_0 = train_0[predictor_features], train_0['BAL_3']
            grid_search_0 = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid, cv=3, scoring='r2', n_jobs=-1
            )
            grid_search_0.fit(X_0, y_0)
            self.regressor_0 = grid_search_0.best_estimator_
            print(f"Class 0 best_params: {grid_search_0.best_params_}")
        else:
            self.regressor_0 = None
            print("No training data for class 0.")

        if not train_1.empty:
            X_1, y_1 = train_1[predictor_features], train_1['BAL_3']
            grid_search_1 = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid, cv=3, scoring='r2', n_jobs=-1
            )
            grid_search_1.fit(X_1, y_1)
            self.regressor_1 = grid_search_1.best_estimator_
            print(f"Class 1 best_params: {grid_search_1.best_params_}")
        else:
            self.regressor_1 = None
            print("No training data for class 1.")

    def reset(self):
        self.current_step = np.random.randint(0, self.n_customers)
        state = self.df.iloc[self.current_step][self.state_space].values
        self.current_l_p = self.df.iloc[self.current_step]['L_P']
        self.current_bal = self.df.iloc[self.current_step]['AVG_BALANCE']
        self.current_int = self.df.iloc[self.current_step]['INT']
        return state

    def step(self, action):
        global provision_bins

        balance_class = self.df.iloc[self.current_step]['BALANCE_CLASS']
        predictor_features = ['BALANCE_CLASS', 'UR_456', 'PR_456', 'INT_456', 'L_P_456', 'BALANCE_MEAN_PAST_456']
        state_data_row = self.df.iloc[self.current_step][predictor_features].values.reshape(1, -1)

        # Apply limit increase if action == 1
        new_l_p = self.current_l_p * (1 + self.beta) if action == 1 else self.current_l_p

        # Predict future balance using regressor
        bal_3_pred = 0
        if balance_class == 0 and self.regressor_0:
            bal_3_pred = max(0, self.regressor_0.predict(state_data_row)[0])
        elif balance_class == 1 and self.regressor_1:
            bal_3_pred = max(0, self.regressor_1.predict(state_data_row)[0])

        pd_value = self.pd_dict[balance_class]

        # Reward = expected interest - expected provision loss
        reward = (
                3 * self.current_int * self.current_bal * (1 - pd_value)
                - pd_value * self.lgd * (bal_3_pred + self.ccf * (new_l_p - bal_3_pred))
        )
        reward = np.clip(reward, -1e6, 1e6) / 1e6

        # Update state: new provision bin
        l_r_value = self.df.iloc[self.current_step]['L_R']
        new_delta_provision = (new_l_p - l_r_value) / l_r_value if l_r_value != 0 else 0
        new_dp_bin_result = pd.cut([new_delta_provision], bins=provision_bins, labels=False, include_lowest=True)
        new_dp_bin = int(new_dp_bin_result[0]) if new_dp_bin_result.size > 0 and not pd.isna(
            new_dp_bin_result[0]) else 0

        new_state = self.df.iloc[self.current_step][self.state_space].values.copy()
        dp_bin_index = self.state_space.index('D_PROVISION_bin')
        new_state[dp_bin_index] = new_dp_bin

        # Move to next customer
        self.current_step = (self.current_step + 1) % self.n_customers
        self.current_l_p = self.df.iloc[self.current_step]['L_P']
        self.current_bal = self.df.iloc[self.current_step]['AVG_BALANCE']
        self.current_int = self.df.iloc[self.current_step]['INT']

        done = self.current_step == 0
        return new_state, reward, done, {}
