from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def average_similarity(vectors):

    similarities = []

    for i in range(len(vectors)):

        for j in range(i + 1, len(vectors)):

            sim = cosine_similarity(
                vectors[i].reshape(1, -1),
                vectors[j].reshape(1, -1)
            )[0][0]

            similarities.append(sim)

    return np.mean(similarities)