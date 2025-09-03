def coherence():
    from load_data import loadPreprocess
    texts, categories, _ = loadPreprocess()

    
    from umap import UMAP
    from hdbscan import HDBSCAN
    from bertopic import BERTopic
    from gensim.models import CoherenceModel
    from gensim.corpora import Dictionary
    # try different parameter combinations and see which ones work best
    parameter_combinations = [(5, 2, 10), (10, 3, 15), (15, 4, 20)]  # neighbors, clusters, components; more aggressive  --> conservative

    # the best one
    retModel = None
    retTopics = None
    retCohScore = -1

    from sentence_transformers import SentenceTransformer
    embedding = SentenceTransformer("all-mpnet-base-v2")
    
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    
    for neighbors, clust_size, comp in parameter_combinations:
        print(f"    neighbors={neighbors}, cluster_size={clust_size}, components={comp}")
        
        # Create models with these parameters
        umap_model = UMAP(n_neighbors=neighbors, n_components=comp, 
                         min_dist=0.0, metric="cosine", random_state=600)
        hdbscan_model = HDBSCAN(min_cluster_size=clust_size, min_samples=1)
        
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model
        )
        
        # step
        topics, probs = topic_model.fit_transform(texts)
        
        # now the coherence score part
    
        topic_words = []
        for topic in set(topics):
            if topic != -1:  # Skip outliers
               # topic representation (tuples)
                topic_representation = topic_model.get_topic(topic)

                # just get the word and ignore the score
                words = []
                for word_score_pair in topic_representation:
                    word = word_score_pair[0] 
                    words.append(word)
                topic_words.append(words)
        
        if len(topic_words) == 0:  
            continue # would be none
        
        # Prepare data for coherence calculation
        tokenized_texts = [text.split() for text in texts]
        dictionary = Dictionary(tokenized_texts)
        
        # Calculate coherence - ripped from reference website
        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence="c_v",
            processes = 1
        )
        coherence_score = coherence_model.get_coherence() # get final score
        
        # best model - keep track
        if coherence_score > retCohScore:
            retCohScore = coherence_score
            retModel = topic_model
            retTopics = topics

    if retModel is None:
        retModel = topic_model
        retTopics = topics
        
    return retModel, retTopics, categories