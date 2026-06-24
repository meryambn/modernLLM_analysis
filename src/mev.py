from sklearn.decomposition import PCA


def compute_mev(vectors):

    pca = PCA(
        n_components=1
    )

    pca.fit(vectors)

    return pca.explained_variance_ratio_[0]