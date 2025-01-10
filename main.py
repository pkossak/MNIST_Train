import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc

# Załadowanie danych MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Podział na zbiór testowy 75% i treningowy 25%
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.25, random_state=13)

# Normalizacja wartości pikseli do zakresu od 0 do 1
x_train, x_val, x_test = x_train / 255.0, x_val / 255.0, x_test / 255.0

# Zdefiniowanie architektury sieci neuronowej
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Warstwa wejściowa — spłaszcza obrazy 28×28
    tf.keras.layers.Dense(256, activation='relu'),   # Warstwa ukryta z 128 neuronami i funkcją aktywacji relu
    tf.keras.layers.Dropout(0.2),                   # Warstwa Dropout w celu regularyzacji
    tf.keras.layers.Dense(10, activation='softmax')  # Warstwa wyjściowa z 10 neuronami i funkcją aktywacji softmax
])


# Zmiana współczynnika uczenia
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Trenowanie modelu
model.compile(optimizer=custom_optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Algorytm wstecznej propagacji błędu
history = model.fit(x_train, y_train, epochs=11, validation_data=(x_val, y_val))

# Wykonaj predykcje na zbiorze testowym przy użyciu wytrenowanego modelu
predictions = model.predict(x_test)

# Konwertuj przewidywane etykiety na postać jednowymiarową
predicted_labels = np.argmax(predictions, axis=1)

# Dokładność, Precyzja, Czułość, Macierz Pomyłek
accuracy = accuracy_score(y_test, predicted_labels)
precision = precision_score(y_test, predicted_labels, average='weighted')
recall = recall_score(y_test, predicted_labels, average='weighted')
conf_matrix = confusion_matrix(y_test, predicted_labels)
print(f'Dokładność: {accuracy:.4f}')
print(f'Precyzja: {precision:.4f}')
print(f'Czułość: {recall:.4f}')
print('Macierz Pomyłek:')
print(conf_matrix)

# Krzywa ROC i pole powierzchni pod krzywą (AUC)
num_classes = 10  # liczba klas w problemie MNIST
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    true_labels = (y_test == i).astype(int)
    pred_scores = predictions[:, i]
    fpr[i], tpr[i], _ = roc_curve(true_labels, pred_scores)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Wyświetlenie krzywych ROC dla poszczególnych klas
plt.figure(figsize=(8, 6))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'Cyfra {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('Krzywa ROC dla każdej cyfry')
plt.legend()
plt.show()
