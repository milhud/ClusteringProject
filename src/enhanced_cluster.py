# basically just keybert for better semantic clustering
def enhanced_representation(sup=True): 

    from load_data import loadPreprocess
    texts, categories, _ = loadPreprocess()

    # encode
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    
    if sup: 
        correct = encoder.fit_transform(categories)
    else:
        correct = None

    from bertopic.representation import KeyBERTInspired
    rep_model = KeyBERTInspired() # more meaningful text


    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("sentence-transformers/all-roberta-large-v1") # larger, more information with more embeddings; 1024 element vectors
    
    from umap import UMAP 
    umap_model = UMAP(n_neighbors=12, n_components=15, min_dist=0.0, metric="cosine", random_state=630) # more information -> need more dimensions to compensate

    from hdbscan import HDBSCAN
    hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=1)  # any point near another can become a core, so more permissive clustering

    from bertopic import BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=rep_model
    )
    
    topics, probs = topic_model.fit_transform(texts, y=correct)
    
    # Return everything needed for evaluation
    return topic_model, topics, categories
