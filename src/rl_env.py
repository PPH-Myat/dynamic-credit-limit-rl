import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

class CreditLimitEnv:
    def __init__(self, df, provision_bins):
        self.df = df.reset_index(drop=True)
        self.df = self.df[self.df['BALANCE_CLASS'].isin([0, 1])].reset_index(drop=True)
        self.provision_bins = provision_bins

        self.df['UR'] = pd.cut(self.df['UR_456'], bins=[-np.inf, 0.25, 0.5, 0.75, 1.0, np.inf], labels=False, include_lowest=True)
        self.df['PR'] = pd.cut(self.df['PR_456'], bins=[-np.inf, 0.25, 0.5, 0.75, 1.0, np.inf], labels=False, include_lowest=True)

        self.state_space = ['BALANCE_CLASS', 'UR', 'PR', 'D_PROVISION_bin']
        self.action_space = [0, 1]
        self.n_customers = len(self.df)
        self.current_step = 0

        self.beta = 0.2
        self.pd_dict = {0: 0.01, 1: 0.05, 2: 0.15}
        self.lgd = 0.5
        self.ccf = 0.8

        self.train_best_regressors()
        self.predict_bal_3_future()

    def train_best_regressors(self):
        self.df['BAL_3_LABEL_456'] = self.df['AVG_BALANCE_456']

        train_df = self.df.copy()
        predictor_features = ['UR_789', 'PR_789', 'INT', 'L_P_789', 'AVG_BALANCE_789']

        self.best_model_0, self.best_model_1 = None, None

        model_grid = {
            'rf': {
                'model': RandomForestRegressor(random_state=42),
                'params': {'n_estimators': [100], 'max_depth': [10], 'max_features': [0.8]}
            },
            'gbr': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [3]}
            }
        }

        for cls in [0, 1]:
            cls_df = train_df[train_df['BALANCE_CLASS'] == cls]
            if cls_df.empty:
                continue

            X, y = cls_df[predictor_features], cls_df['BAL_3_LABEL_456']
            best_model, best_score = None, -np.inf

            for cfg in model_grid.values():
                grid = GridSearchCV(cfg['model'], cfg['params'], cv=3, scoring='r2', n_jobs=-1)
                grid.fit(X, y)
                if grid.best_score_ > best_score:
                    best_score = grid.best_score_
                    best_model = grid.best_estimator_

            if cls == 0:
                self.best_model_0 = best_model
            else:
                self.best_model_1 = best_model

    def predict_bal_3_future(self):
        predictor_features = ['UR_456', 'PR_456', 'INT', 'L_P', 'AVG_BALANCE_456']
        preds = []

        for _, row in self.df.iterrows():
            cls = row['BALANCE_CLASS']
            X_row = row[predictor_features].values.reshape(1, -1)
            model = self.best_model_0 if cls == 0 else self.best_model_1
            pred = model.predict(X_row)[0] if model else 0.0
            preds.append(pred)

        self.df['BAL_3_pred'] = preds

    def reset(self):
        self.current_step = np.random.randint(0, self.n_customers)
        row = self.df.iloc[self.current_step]
        return row[self.state_space].values

    def step(self, action):
        row = self.df.iloc[self.current_step]

        # --- Simulated Action Outcome ---
        new_l_p = row['L_P'] * (1 + self.beta) if action == 1 else row['L_P']
        bal_3_pred = row['BAL_3_pred']
        pd_value = self.pd_dict[row['BALANCE_CLASS']]

        # --- Simulated Reward ---
        reward = (
                3 * row['INT'] * row['AVG_BALANCE_456'] * (1 - pd_value)
                - pd_value * self.lgd * (bal_3_pred + self.ccf * (new_l_p - bal_3_pred))
        )
        reward = np.clip(reward, -1e6, 1e6) / 1e6

        # --- For Discretization of Next State ---
        delta_prov = (new_l_p - row['L_R']) / row['L_R'] if row['L_R'] != 0 else 0
        new_dp_bin = pd.cut([delta_prov], bins=self.provision_bins, labels=False, include_lowest=True)
        new_dp_bin = int(new_dp_bin[0]) if new_dp_bin.size > 0 and not pd.isna(new_dp_bin[0]) else 0

        # --- Next State ---
        new_state = row[self.state_space].values.copy()
        new_state[self.state_space.index('D_PROVISION_bin')] = new_dp_bin

        # --- True Ground-Truth for Offline Evaluation ---
        actual_balance = row['AVG_BALANCE_123']  # <-- must exist in df
        info = {
            'actual_balance': actual_balance,
            'interest_rate': row['INT'],
            'pd': pd_value,
            'lgd': self.lgd,
            'new_limit': new_l_p,
            'ccf': self.ccf
        }

        # --- Step Forward ---
        self.current_step = (self.current_step + 1) % self.n_customers
        done = self.current_step == 0

        return new_state, reward, done, info
