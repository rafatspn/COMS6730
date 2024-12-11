import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import random

# Step 1: Train Target Model (Black-box)
def train_target_model(X, y):
    """ Simulate a black-box target model """
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

# Step 2: Shadow Model Training
def train_shadow_models(X_pool, y_pool, num_shadow=5, train_size=0.5):
    """ Train shadow models on disjoint datasets """
    shadow_models = []
    shadow_data = []
    for _ in range(num_shadow):
        X_train, X_test, y_train, y_test = train_test_split(X_pool, y_pool, train_size=train_size)
        shadow_model = RandomForestClassifier(n_estimators=50)
        shadow_model.fit(X_train, y_train)
        shadow_models.append(shadow_model)
        shadow_data.append((X_train, X_test))  # Store train and test data
    return shadow_models, shadow_data

# Step 3: Query Shadow Models to Create Attack Dataset
def create_attack_dataset(shadow_models, shadow_data):
    """ Generate training data for attack model (in/out labels) """
    attack_X, attack_y = [], []
    for shadow_model, (X_in, X_out) in zip(shadow_models, shadow_data):
        # Query shadow model with 'in' data (used during shadow training)
        for x in X_in:
            pred = shadow_model.predict_proba([x])[0]
            attack_X.append(pred)
            attack_y.append(1)  # Label 1: Member
        # Query shadow model with 'out' data (not seen during training)
        for x in X_out:
            pred = shadow_model.predict_proba([x])[0]
            attack_X.append(pred)
            attack_y.append(0)  # Label 0: Non-Member
    return np.array(attack_X), np.array(attack_y)

# Step 4: Train the Attack Model
def train_attack_model(attack_X, attack_y):
    """ Train attack model to predict membership inference """
    attack_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
    attack_model.fit(attack_X, attack_y)
    return attack_model

# Step 5: Membership Inference Attack
def perform_attack(target_model, attack_model, X_in, X_out):
    """ Use the attack model to infer membership on target model's data """
    attack_input = []
    for x in X_in:
        pred = target_model.predict_proba([x])[0]
        attack_input.append(pred)
    attack_preds_in = attack_model.predict(attack_input)

    attack_input = []
    for x in X_out:
        pred = target_model.predict_proba([x])[0]
        attack_input.append(pred)
    attack_preds_out = attack_model.predict(attack_input)

    return attack_preds_in, attack_preds_out

# Example Usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris

    # Load Dataset
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_pool, y_train, y_pool = train_test_split(X, y, test_size=0.5, random_state=42)

    # 1. Train Target Model
    target_model = train_target_model(X_train, y_train)

    # 2. Train Shadow Models
    shadow_models, shadow_data = train_shadow_models(X_pool, y_pool, num_shadow=3)

    # 3. Create Attack Dataset
    attack_X, attack_y = create_attack_dataset(shadow_models, shadow_data)

    # 4. Train Attack Model
    attack_model = train_attack_model(attack_X, attack_y)

    # 5. Perform Membership Inference Attack
    attack_preds_in, attack_preds_out = perform_attack(target_model, attack_model, X_train, X_pool)

    # Evaluate the Attack
    print("Attack Accuracy on Members:", accuracy_score([1] * len(X_train), attack_preds_in))
    print("Attack Accuracy on Non-Members:", accuracy_score([0] * len(X_pool), attack_preds_out))
