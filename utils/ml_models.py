import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons, make_circles, make_classification

def get_dataset(name, n_samples=300):
    if name == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=0.15, random_state=42)
    elif name == "Circles":
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
    elif name == "Linear":
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
    elif name == "XOR":
        rng = np.random.RandomState(42)
        X = rng.randn(n_samples, 2)
        y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    else:
        # Default
        X, y = make_moons(n_samples=n_samples, noise=0.15, random_state=42)
    
    return X, y

def train_mlp(X, y, activation='relu', hidden_layer_sizes=(16, 16), max_iter=200, learning_rate_init=0.01):
    # Map friendly names to sklearn names if possible
    act_map = {
        'ReLU': 'relu',
        'Tanh': 'tanh',
        'Sigmoid': 'logistic',
        'Identity (Linear)': 'identity'
    }
    
    sk_activation = act_map.get(activation, 'relu')
    
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=sk_activation,
        solver='adam',
        max_iter=max_iter,
        learning_rate_init=learning_rate_init,
        random_state=42
    )
    
    model.fit(X, y)
    return model

def create_decision_boundary_data(model, X, grid_resolution=100):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
                         np.linspace(y_min, y_max, grid_resolution))
    
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    return xx, yy, Z
