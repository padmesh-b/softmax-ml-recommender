import numpy as np

def softmax(x):
    x = np.array(x)
    exp_x = np.exp(x - np.max(x))  # stability
    return exp_x / np.sum(exp_x)

def recommend(user_vector, item_vectors, item_names):
    scores = np.dot(item_vectors, user_vector)

    probs = softmax(scores)

    # probabilistic recommendation (more real-world)
    recommended_index = np.argmax(probs)

    return scores, probs, item_names[recommended_index]