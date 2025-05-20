import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


print("Carregando dataset CIFAR-10...")
cifar10 = fetch_openml('cifar_10', version=1, cache=True)
X = cifar10.data
y = cifar10.target.astype(np.uint8)

print(f"Shape dos dados: {X.shape}")
print(f"Shape dos rótulos: {y.shape}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Treina o modelo Decision Tree
print("\nTreinando modelo Decision Tree...")
start_time = time.time()

tree = DecisionTreeClassifier(max_depth=20, random_state=42)
tree.fit(X_train, y_train)

train_time = time.time() - start_time
print(f"Tempo de treinamento: {train_time:.2f} segundos")

# Avaliação do modelo
print("\nAvaliando modelo...")
start_time = time.time()

y_pred = tree.predict(X_test)

test_time = time.time() - start_time
print(f"Tempo de predição: {test_time:.2f} segundos")

# Relatório de classificação
print("\nDecision Tree - Classification Report")
print(classification_report(y_test, y_pred))

# Matriz de confusão
ConfusionMatrixDisplay.from_estimator(
    tree, 
    X_test, 
    y_test,
    cmap='Greens',
    values_format='d',
    display_labels=[str(i) for i in range(10)]
)
plt.title("Matriz de Confusão - Decision Tree (max_depth=20)")
plt.tight_layout()
plt.savefig('confusion_matrix_tree.png')
plt.show()