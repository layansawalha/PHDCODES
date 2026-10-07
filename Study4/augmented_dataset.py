import random

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

df = pd.read_csv('augmented_dataset_final.csv')
df.drop(columns=['contract_contract_sum', 'cost_increment'], inplace=True, errors='ignore')
df.dropna(inplace=True)

target = 'cost_rebased'

required_cols = {'record_id', 'is_augmented', 'source_record_id'}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(
        f"Missing columns required for the train-test split: {sorted(missing)}."
    )

original_df = df[df['is_augmented'] == 0]
augmented_df = df[df['is_augmented'] == 1]

original_train_df, original_test_df = train_test_split(
    original_df, test_size=0.2, random_state=SEED,
)
train_ids = set(original_train_df['record_id'])
test_ids = set(original_test_df['record_id'])

leaked = augmented_df[augmented_df['source_record_id'].isin(test_ids)]
if not leaked.empty:
    raise RuntimeError(
        f"{len(leaked)} augmented rows are derived from test-set records."
    )
augmented_train_df = augmented_df[augmented_df['source_record_id'].isin(train_ids)]

train_df = pd.concat([original_train_df, augmented_train_df], ignore_index=True)
test_df = original_test_df.reset_index(drop=True)

drop_cols = [target, 'record_id', 'is_augmented', 'source_record_id']
X_train = train_df.drop(columns=drop_cols, errors='ignore')
y_train = train_df[target]
X_test  = test_df.drop(columns=drop_cols, errors='ignore')
y_test  = test_df[target]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                   random_state=SEED)
xgb.fit(X_train, y_train)
xgb_train_features = xgb.predict(X_train).reshape(-1, 1)
xgb_test_features = xgb.predict(X_test).reshape(-1, 1)

cat = CatBoostRegressor(verbose=0, iterations=300, learning_rate=0.05,
                        depth=6, random_state=SEED)
cat.fit(X_train, y_train)
cat_train_features = cat.predict(X_train).reshape(-1, 1)
cat_test_features = cat.predict(X_test).reshape(-1, 1)

mlp_input = Input(shape=(X_train_scaled.shape[1],), name='mlp_input')
x = Dense(128, activation='relu')(mlp_input)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
mlp_output = Dense(32, activation='relu')(x)

xgb_input = Input(shape=(1,), name='xgb_input')
cat_input = Input(shape=(1,), name='cat_input')

merged = Concatenate()([mlp_output, xgb_input, cat_input])
z = Dense(64, activation='relu')(merged)
z = Dropout(0.3)(z)
z = Dense(32, activation='relu')(z)
final_output = Dense(1, activation='linear')(z)

model = Model(inputs=[mlp_input, xgb_input, cat_input], outputs=final_output)
model.compile(optimizer='adam', loss='mse',
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

early_stop = EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(
    [X_train_scaled, xgb_train_features, cat_train_features],
    y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1,
)

y_pred = model.predict([X_test_scaled, xgb_test_features, cat_test_features]).flatten()
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
cv_rmse = rmse / np.mean(y_test) * 100
nrmse = rmse / (np.max(y_test) - np.min(y_test)) * 100
print("\n--- Hybrid MLP + XGBoost + CatBoost Results ---")
print(f"Test R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE (%): {mape:.2f}")
print(f"CV(RMSE) (%): {cv_rmse:.2f}")
print(f"NRMSE (%): {nrmse:.2f}")
