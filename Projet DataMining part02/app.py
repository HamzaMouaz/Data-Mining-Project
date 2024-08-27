import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from itertools import combinations
from collections import defaultdict
from collections import Counter
from sklearn.metrics import silhouette_score
import seaborn as sns
import random
from pprint import pprint
from sklearn.model_selection import train_test_split


with open('dataset_prétraitement.csv', 'r') as file:
    data = [line.strip().split(',') for line in file.readlines() ]
    
def convert_to_float(dataset):
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            try:
                dataset[i][j] = float(dataset[i][j])
            except ValueError:
                # Gérer les valeurs qui ne peuvent pas être converties en flottant
                pass  # Vous pouvez choisir de remplacer par une valeur par défaut ou de gérer autrement
    return dataset
data=convert_to_float(data)
 
X=[]
y=[]
# Séparation des caractéristiques (X) et de la cible (y)
for d in data:
 X.append(d[:-1])  # Caractéristiques
 y.append(d[-1])  # Attribut cible à prédire
y=y[1:]
X=X[1:]
for t in y:
    t=int(t)


# Séparation des données en ensembles d'apprentissage et de test (80% train / 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

def euclidean_distance(instance1, instance2):
            
  return np.sqrt(np.sum((instance1 - instance2) ** 2))

def sort_instances_by_distance(instance,X_train,y_train, distance_function):
    distances = [(index, distance_function(instance, X_train)) for index, X_train in enumerate(X_train)]
    distances.sort(key=lambda x: x[1])
    return distances
instance = np.array([138.0,8.6,560.0,7.46,0.62,0.7,5.9,0.24,0.31,0.77,8.71,0.11,1.204])
dis = sort_instances_by_distance(instance,X_train,y_train, euclidean_distance)


def get_majority_class(classes):
  compteur_classes = Counter(classes)
    
    # Obtenir la classe dominante (la plus fréquente)
  classe_dominante = compteur_classes.most_common(1)[0][0]
  return classe_dominante

def k_nearest_neighbors(k, Inst,X_train,y_train ,distance_function):
    distances = sort_instances_by_distance(Inst,X_train,y_train, distance_function)
    knn = distances[:k]
    
    k_nearest_indices = [knns[0] for knns in knn]
    classe=[]
    
    for v in k_nearest_indices:
        classe.append(y_train[v])
    #print(k_nearest_indices)    
    #print(classe)
    return  get_majority_class(classe)
#DECISION TREES
df = pd.read_csv("dataset_prétraitement.csv")
#df = df.drop("Id", axis=1)
df = df.rename(columns={"Fertility": "label"})

def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df

random.seed(0)
train_df, test_df = train_test_split(df, test_size=20)

def check_purity(data):
    
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False
def classify_data(data):
    
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification

def get_potential_splits(data):
    
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):        # excluding the last column which is the label
        potential_splits[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)

        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_split = (current_value + previous_value) / 2
                
                potential_splits[column_index].append(potential_split)
    
    return potential_splits

def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]

    data_below = data[split_column_values <= split_value]
    data_above = data[split_column_values >  split_value]
    
    return data_below, data_above

def calculate_entropy(data):
    
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy
def calculate_overall_entropy(data_below, data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy =  (p_data_below * calculate_entropy(data_below) 
                      + p_data_above * calculate_entropy(data_above))
    
    return overall_entropy
def determine_best_split(data, potential_splits):
    
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value
def decision_tree_algorithm(df, counter=0,max_depth=5):
    
    # data preparations
    if counter == 0:
        data = df.values
    else:
        data = df           
    
    
    # base cases
    if check_purity(data) or (counter == max_depth):
        classification = classify_data(data)
        return classification

    
    # recursive part
    else:    
        counter += 1

        # helper functions 
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        
        # instantiate sub-tree
        question = "{} <= {}".format(split_column, split_value)
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter)
        no_answer = decision_tree_algorithm(data_above, counter)
        
        sub_tree[question].append(yes_answer)
        sub_tree[question].append(no_answer)
        
        return sub_tree

def predict(sample,tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split()

    
    if sample[int(feature_name)] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]

    # Si la réponse est un dictionnaire, cela signifie qu'il y a plus de questions à poser
    if not isinstance(answer, dict):
        return answer  # C'est une classification
    else:
        # Récursion pour continuer la construction de l'arbre
        residual_tree = answer
        return predict(sample,residual_tree)

def bootstrapping(train_df, n_bootstrap):
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]
    
    return df_bootstrapped
def bootstrapping1(train_df, n_bootstrap, n_features):
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    selected_features = np.random.choice(train_df.columns[:-1], size=n_features, replace=False)
    
    # Création d'une copie du DataFrame train_df sans les colonnes sélectionnées
    train_df_without_selected = train_df.drop(columns=selected_features)
    
    # Création de l'échantillon bootstrap en utilisant les indices bootstrap sur le DataFrame modifié
    df_bootstrapped = train_df_without_selected.iloc[bootstrap_indices]
    
    return df_bootstrapped

def random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
    forest = []
    for i in range(n_trees):
        df_bootstrapped =  bootstrapping1(train_df, n_bootstrap, n_features)
        tree = decision_tree_algorithm(df_bootstrapped)
        forest.append(tree)
    
    return forest

def random_forest_predictions(test_df, forest):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        ftree=forest[i]
        predictions = predict(test_df,ftree)
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]
    
    return random_forest_predictions


st.set_page_config(page_title="my interface",page_icon=":tada:",layout="wide")
with st.container():
 st.subheader("WELCOME :wave: :wave:")
 st.title("Projet DATA MINING")


st.title('Choix de méthode')

method = st.radio('Choisissez une méthode :', ('KNN', 'Decision Trees', 'Random Forest'))

if method == 'KNN':
    neighbors = st.text_input('Nombre de voisins')
    #distance = st.text_input('Distance')
    soil_data = st.text_input('Enter soil data (N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B, OM)', value='138.0,8.6,560.0,7.46,0.62,0.7,5.9,0.24,0.31,0.77,8.71,0.11,1.204')
    

elif method == 'Decision Trees':
    depth = st.text_input('Profondeur de l\'arbre')
    soil_data = st.text_input('Enter soil data (N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B, OM)', value='138.0,8.6,560.0,7.46,0.62,0.7,5.9,0.24,0.31,0.77,8.71,0.11,1.204')

elif method == 'Random Forest':
    num_trees = st.text_input('Nombre d\'arbres')
    depth = st.text_input('Profondeur')
    soil_data = st.text_input('Enter soil data (N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B, OM)', value='138.0,8.6,560.0,7.46,0.62,0.7,5.9,0.24,0.31,0.77,8.71,0.11,1.204')

st.write("Vous avez choisi la méthode :", method)
if method == 'KNN':
    st.write("Nombre de voisins :", neighbors)
    #st.write("Distance :", distance)
    input_data = np.array(soil_data.split(','), dtype=float)
    
elif method == 'Decision Trees':
    st.write("Profondeur de l'arbre :", depth)
    input_data = np.array(soil_data.split(','), dtype=float)
elif method == 'Random Forest':
    st.write("Nombre d'arbres :", num_trees)
    st.write("Profondeur :", depth)
    input_data = np.array(soil_data.split(','), dtype=float)
if st.button('Predict'):
   if method=='KNN':
    result_k3 = k_nearest_neighbors(int(neighbors), input_data,X_train,y_train, euclidean_distance)
    st.write("prediction : classe", result_k3)
   elif method=='Decision Trees':
       tree = decision_tree_algorithm(train_df)
       predicted_class = predict(input_data,tree)
       st.write("prediction : classe", predicted_class)
   elif method=='Random Forest':
       forest = random_forest_algorithm(train_df, n_trees=int(num_trees), n_bootstrap=800, n_features=9, dt_max_depth=int(depth))
       predictions=[]
       for g in range(len(forest)):
          tree=forest[g]
 
 
          prediction= predict(input_data, tree)
          predictions.append(prediction)
          counter = Counter(predictions)

# Trouver l'élément le plus fréquent
       most_common_element = counter.most_common(1)[0][0]
       st.write("prediction : classe", most_common_element)
       
      

       
   





  



