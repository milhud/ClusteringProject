def ensemble():

    from load_data import loadPreprocess
    texts, categories, _ = loadPreprocess()
    
    # run multiple models with different parameters
    models = []
    all_topics = []
    
    # imports 
    from sentence_transformers import SentenceTransformer
    from hdbscan import HDBSCAN
    from umap import UMAP
    from bertopic import BERTopic

    # first embeddings
    emb1 = SentenceTransformer("all-mpnet-base-v2")
    umap1 = UMAP(n_neighbors=5, n_components=10, min_dist=0.1, random_state=158) # localized; 5 nearest neighbors; looser clusters
    hdb1 = HDBSCAN(min_cluster_size=3, min_samples=2) # somewhat conservative in cluster formation vs outlier
    model1 = BERTopic(embedding_model=emb1, umap_model=umap1, hdbscan_model=hdb1)
    topics1, _ = model1.fit_transform(texts)
    models.append(model1)
    all_topics.append(topics1)
    
    # this is more aggressive clustering
    emb2 = SentenceTransformer("sentence-transformers/all-roberta-large-v1")
    umap2 = UMAP(n_neighbors=15, n_components=20, min_dist=0.0) # aggressive clustering, 20 dimensions -> 15 nearest neighbors while keeping more info
    hdb2 = HDBSCAN(min_cluster_size=2, min_samples=1) # smaller clusters
    model2 = BERTopic(embedding_model=emb2, umap_model=umap2, hdbscan_model=hdb2)
    topics2, _ = model2.fit_transform(texts)
    models.append(model2)
    all_topics.append(topics2)
    
    # take the model with fewer outliers
    import numpy as np
    outlier_counts = []
    for topics in all_topics:
        outlier_count = 0
        for t in topics:
            if t == -1:
                outlier_count = outlier_count + 1
        
        outlier_counts.append(outlier_count)
    
    best_idx = np.argmin(outlier_counts)
    
    return models[best_idx], all_topics[best_idx], categories