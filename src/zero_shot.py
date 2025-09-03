import multiprocessing
def zero_shot():
    from load_data import loadPreprocess
    texts, categories, _ = loadPreprocess()

    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN

    topic_list = list(set(categories))
    embedding = SentenceTransformer("all-mpnet-base-v2")
    
    # Add proper parameters to handle small datasets
    umap_model = UMAP(n_neighbors=min(15, len(texts)-1), n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=1)
    
    mod = BERTopic(
        embedding_model=embedding, 
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        zeroshot_topic_list=topic_list, 
        zeroshot_min_similarity=0.3
    )

    topics, probs = mod.fit_transform(texts)

    return mod, topics, categories