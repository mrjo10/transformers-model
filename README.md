# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping
import pickle

import warnings
sns.set_theme(context='notebook', style='whitegrid')
warnings.filterwarnings('ignore')
data = pd.read_csv('Modified_Pizza_Sales_Data.csv')
# Initial preprocessing
data['order_date'] = pd.to_datetime(data['order_date'])
data['year'] = data['order_date'].dt.year
data['month'] = data['order_date'].dt.month
data['day'] = data['order_date'].dt.day
data['day_of_week'] = data['order_date'].dt.day_name()
data['total_sales'] = data['total_price'] * data['quantity']

# Aggregating daily sales
daily_sales = data.groupby(['order_date', 'day_of_week']).agg({'total_sales': 'sum'}).reset_index()

# Splitting into features and labels
X = daily_sales[['order_date', 'day_of_week']]
y = daily_sales['total_sales']
# One-hot encode categorical variables
encoder = OneHotEncoder()
days_encoded = encoder.fit_transform(X[['day_of_week']]).toarray()
days_columns = encoder.get_feature_names_out(['day_of_week'])
days_encoded_df = pd.DataFrame(days_encoded, columns=days_columns)

# Combine encoded days with other features
X = pd.concat([X.reset_index(drop=True), days_encoded_df], axis=1)
X['days_since_start'] = (X['order_date'] - X['order_date'].min()).dt.days
X = X.drop(['day_of_week', 'order_date'], axis=1)
# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Define Transformer-based model
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
# Define input layer
input_layer = Input(shape=(X_train.shape[1],))
embed_dim = 128
num_heads = 4
ff_dim = 128

# Expand dimensions for transformer compatibility
x = tf.expand_dims(input_layer, axis=1)

# Pass through embedding and transformer layers
x = Dense(embed_dim, activation="relu")(x)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)

# Flatten the sequence dimension
x = tf.squeeze(x, axis=1)

# Dense layers for final output
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(32, activation="relu")(x)
output_layer = Dense(1, activation="linear")(x)

# Compile the model
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
# Train the model 
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)
# Step 4: Evaluate the Model
loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict(X_test)
r2 = 1 - np.sum((y_test - y_pred.flatten())**2) / np.sum((y_test - np.mean(y_test))**2)


print("\nModel Performance:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")


plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show() 
