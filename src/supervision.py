from load_data import loadPreprocess # for the loadPreprocess function; returns texts, categories, and cleaned questions only

def basic_supervision(sup=True):
    texts, categories, _ = loadPreprocess()
    
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    
    if sup:
        correct = encoder.fit_transform(categories) # corret[0] -> correct for q 0
    else:
        correct = None

    embedding_model = SentenceTransformer("all-mpnet-base-v2") #

    # pre trained model that is better for technical content    
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    # reduce dimensions for better clustering
    from umap import UMAP
    umap_model = UMAP(n_neighbors=8, n_components=10, min_dist=0.0, metric="cosine", random_state=42) # random set to reproduce, cosine for angle between instead of magnitude

    from hdbscan import HDBSCAN
    hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=1, metric="euclidean") # close two points to make a cluster

    # finally run model
    from bertopic import BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
    )

    # embedding (to vector) -> umap -> clustering -> uses correct to guide clustering -> extracts topics
    topics, probs = topic_model.fit_transform(texts, y=correct)

    return topic_model, topics, categories,