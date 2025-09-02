def iterative(sup=True):

    from load_data import loadPreprocess
    texts, categories, _ = loadPreprocess()
    
    # supervised first
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()

    if sup:
        correct = encoder.fit_transform(categories)
    else:
        correct = None
    
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from bertopic import BERTopic
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    umap_model = UMAP(n_neighbors=10, n_components=15, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=1)
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model
    )
    
    # run first
    topics, probs = topic_model.fit_transform(texts, y=correct)
    
    # then process outliers
    original_outliers = sum(1 for t in topics if t == -1)
    
    if original_outliers > 0:
        # loop to lower threshold more and more
        topics = topic_model.reduce_outliers(texts, topics, strategy="embeddings", threshold=0.1)
        new_outliers = sum(1 for t in topics if t == -1)
    
    return topic_model, topics, categories