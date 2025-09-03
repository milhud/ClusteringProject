def bayesian():

    from load_data import loadPreprocess
    texts, categories, _ = loadPreprocess()
   
    from evaluate_clustering import evaluate_clustering_detailed

    from skopt import gp_minimize
    from skopt.space import Integer, Real
    parameters  = [Integer(3,20,name="neighbors"), Integer(2,8,name="min_cluster"), Integer(5,25,name="comps"), Real(0.0,0.3, name="dist")]

    from sentence_transformers import SentenceTransformer
    embedding = SentenceTransformer("all-mpnet-base-v2")

    from umap import UMAP
    from hdbscan import HDBSCAN
    from bertopic import BERTopic

    def objective_function(params): # called by optimization function
        neighbors, min_clust, comps, dist = params

        
        umap_model = UMAP(n_neighbors=neighbors, n_components=comps, min_dist=dist, metric="cosine", random_state=360)
        hdbscan_model = HDBSCAN(min_cluster_size=min_clust, min_samples=1)
        topic_model = BERTopic(embedding_model=embedding, umap_model=umap_model, hdbscan_model=hdbscan_model)

        topics, _ = topic_model.fit_transform(texts)
        ari, nmi, success_rate = evaluate_clustering_detailed(topics, categories)
        return -ari  # we have to optimize which gives maximum, so we need to reverse to get minimum
    
    # Run Bayesian optimization
    result = gp_minimize(func=objective_function, dimensions=parameters, n_calls=15, n_initial_points=5, random_state=42)
    
    # Get best parameters and build final model
    best_n_neighbors, best_min_cluster_size, best_n_components, best_min_dist = result.x
    
    # Reference BERTopic Docsc: https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html#n_neighbors
    retUmap = UMAP(n_neighbors=best_n_neighbors, n_components=best_n_components, min_dist=best_min_dist, metric="cosine")
    retHdbscan = HDBSCAN(min_cluster_size=best_min_cluster_size, min_samples=1)
    retModel = BERTopic(embedding_model=embedding, umap_model=retUmap, hdbscan_model=retHdbscan)
    
    retTopics, _ = retModel.fit_transform(texts)
    
    return retModel, retTopics, categories