from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Lo que primero se hace es obtener el dataset de Enfermedad Renal Crónica, luego de importar el repo de Ucirepo
chronic_kidney_disease = fetch_ucirepo(id=336)

# Luego accedemos a los datos y variables objetivo
X = chronic_kidney_disease.data.features
y = chronic_kidney_disease.data.targets

# convertimos a DataFrame para facilitar los temas de manipulación de datos
X = pd.DataFrame(X, columns=chronic_kidney_disease.variables['name'][:-1])

# Acá busqué de que 'y' sea una Serie de una sola dimensión
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]  # Seleccionar la primera columna si 'y' es un DataFrame
else:
    y = pd.Series(y.ravel(), name='class')  # Aseguramos que 'y' sea una Serie

# Luego imprimimos los metadatos y variables para la revisión respectiva
print(chronic_kidney_disease.metadata)
print(chronic_kidney_disease.variables)

# Verificacamos la forma del 'y' definido previamente
print("Forma de y:", y.shape)

# Imputamos los valores faltantes mediante la estrategia más frecuente
imputer = SimpleImputer(strategy="most_frequent")
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Luego identificamos las columnas categóricas
categorical_columns = X.select_dtypes(include=['object']).columns

# acá aplicamos lo que es LabelEncoder a las columnas categóricas específicamente
for col in categorical_columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# programamos  la variable objetivo
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Acá verificamos si el dataset está desbalanceado
print(y.value_counts())

# En caso que si, realizamos el sobremuestreo de la clase minoritaria
if y.value_counts().min() / y.value_counts().max() < 0.5:
    # acá vamos a combinar las características y etiquetas para realizar el re-muestreo
    data = pd.concat([X, y], axis=1)
    majority_class = data[data['class'] == y.value_counts().idxmax()]
    minority_class = data[data['class'] == y.value_counts().idxmin()]
    
    # acá realizo el sobremuestreo de la clase minoritaria
    minority_oversampled = resample(minority_class, 
                                    replace=True,    # Esta es la muestra con reemplazo
                                    n_samples=len(majority_class), # Se procede a igualar la clase mayoritaria
                                    random_state=42)
    data_balanced = pd.concat([majority_class, minority_oversampled])
    
    # acá lo que hice fue separar de nuevo en características y etiquetas
    X = data_balanced.drop('class', axis=1)
    y_encoded = encoder.fit_transform(data_balanced['class'])

# División del dataset en los conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Se define los modelos y parámetros para afinamiento
models = {
    'Decision Tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [100, 200],
            'max_features': ['sqrt', 'log2']  
        }
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(algorithm='SAMME'),  
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1]
        }
    }
}

# Se entrena y evalúa los modelos
for model_name, mp in models.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5)
    clf.fit(X_train, y_train)
    print(f"Mejores parámetros para {model_name}: {clf.best_params_}")
    
    # acá hice la validación cruzada
    scores = cross_val_score(clf.best_estimator_, X_train, y_train, cv=5)
    print(f"Puntuación promedio para {model_name}: {scores.mean()}")
    
    # Se evalúa finalmente en el conjunto de prueba
    y_pred = clf.predict(X_test)
    print(f"Reporte de clasificación para {model_name}:\n{classification_report(y_test, y_pred)}\n")
    
    # se genera la matriz de confusión y imprime la imagen
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión para {model_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Actual')
    plt.show()
    
    # Acá sale la grafica de barras para la precisión, recall y f1-score, en formato de imagen igualmente
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    df_report[['precision', 'recall', 'f1-score']].iloc[:-1, :].plot(kind='bar')
    plt.title(f'Métricas para {model_name}')
    plt.ylim(0, 1)
    plt.show()

# Acá para poder visualizar la distribución de clases
class_counts = y.value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.title('Distribución de las clases en el dataset original')
plt.xlabel('Clases')
plt.ylabel('Número de instancias')
plt.savefig('clase_distribucion.png')
plt.show()
