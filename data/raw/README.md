# Raw Data Folder

This folder contains the **original data files** from the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) Kaggle competition.

These CSV files include:

- `application_train.csv`: Main training data with labeled outcomes (`TARGET`)
- `application_test.csv`: Test set without labels
- `bureau.csv`: Bureau credit history from other financial institutions
- `bureau_balance.csv`: Monthly status records for each loan in `bureau.csv`
- `credit_card_balance.csv`: Credit card statement-level history for customers
- `previous_application.csv`: All past loan applications (approved/rejected)

> **Note**: These raw `.csv` files are excluded from Git version control via `.gitignore`.