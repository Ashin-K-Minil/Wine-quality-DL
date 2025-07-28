import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical #type: ignore
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense #type: ignore

df = pd.read_csv("WineQT.csv")

print(df.info())
print(df.describe())

# Check for null values
print(df.isnull().sum())

# Handling outliers
for i in df.columns:
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.boxplot([df[i]])
    plt.title("Before outliers")
    plt.xticks([1],labels=[i])

    # IQR method to remove outliers
    if i not in ['quality','Id']:
        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[i] >= lower_bound) & (df[i] <= upper_bound)]

    plt.subplot(1,2,2)
    plt.boxplot([df[i]])
    plt.title("After outliers")
    plt.xticks([1],labels=[i])

    plt.tight_layout()
    plt.show()

# Dropping unnecessary columns and resetting the index
df.reset_index(inplace=True, drop=True)
df.drop(['Id'], axis= 1, inplace=True)

# Initializing features and target variables
X = df.drop('quality', axis=1)
y = df['quality'].values # NN takes numpy arrays as input
y = to_categorical(y)

# Scaling the values to the same range
scaler = StandardScaler()
scaled_values = scaler.fit_transform(X)
# Not required as we use numpy array to train NN
X = pd.DataFrame(scaled_values, columns=X.columns)
X = X.values # NN takes numpy arrays as input

# Splitting training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the NN model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape = (X.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(y.shape[1], activation='Softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)

y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print(y_pred[0])
print(y_pred_classes[0])

print("Confusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes))