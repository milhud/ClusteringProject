def zero_shot():
    from load_data import loadPreprocess
    texts, categories, _ = loadPreprocess()

    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic

    top_list = list(set(categories))
    embedding = SentenceTransformer("all-mpnet-base-v2")
    mod = BERTopic(embedding_model=embedding, zeroshot_topic_list=top_list, zeroshot_min_similarity=0.3)

    topics, probs = mod.fit_transform(texts)

    return mod, topics, categories
   