import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

# Load dataset
data = pd.read_csv('preprocessedfinal_mental_health_data_standardized.csv')

# Drop non-numeric columns and target
X = data.drop(columns=['index', 'Entity', 'Code', 'Year', 'Depression (%)'])
y = (data['Depression (%)'] > data['Depression (%)'].median()).astype(int)

# Feature Engineering
X['Schizophrenia_Bipolar'] = X['Schizophrenia (%)'] * X['Bipolar disorder (%)']
X['Anxiety_Drug'] = X['Anxiety disorders (%)'] * X['Drug use disorders (%)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the classes in the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Define the LightGBM model (initial parameters)
lgb_model = lgb.LGBMClassifier(
    boosting_type='gbdt',
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.1,
    n_estimators=50,
    random_state=42
)
