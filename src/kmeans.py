def kmeans():

    from load_data import loadPreprocess
    texts, categories, _ = loadPreprocess()
    all_clusters = len(set(categories)) # all clusters
    
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("all-mpnet-base-v2") # better for technical terms
    
    from umap import UMAP
    umap_model = UMAP(n_neighbors=15, n_components=20, min_dist=0.0, metric="cosine", random_state=999) # angle instead of euclidean
    
    from hdbscan import HDBSCAN
    from sklearn.cluster import KMeans
    kmeans_model = KMeans(n_clusters=all_clusters, n_init=10) # ten times ten clusters
    
    from bertopic import BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=kmeans_model # enforces no outliers but otherwise works
    )
    
    topics, probs = topic_model.fit_transform(texts)
    return topic_model, topics, categories
