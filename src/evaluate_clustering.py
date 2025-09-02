def evaluate_clustering_detailed(topics, categories): # ground truth categories passed
    
    # stats
    unique_topics = set(topics)
    clusters = len(unique_topics) # includes outliers
    outliers = 0

    for topic in topics:
        if topic == -1:
            outliers += 1
    
    clustered = len(topics) - outliers
    outlier_percentage = (outliers / len(topics)) * 100
    clustered_percentage = (clustered / len(topics)) * 100
    
    # if all outliers just return with zeroes
    if clustered == 0:
        return 0.0, 0.0, 0.0
    
    # going to start by eliminating the outliers - first create mask
    mask = []
    for topic in topics:
        if topic != -1:
            mask.append(True)
        else:
            mask.append(False)
    
    # filter with mask - keep non outliers
    filtered_topics = []
    for i in range(len(topics)):
        topic = topics[i]
        keep = mask[i]
        if keep:
            filtered_topics.append(topic)
    
    # now the same for the categories
    filtered_categories = []
    for i in range(len(categories)):
        category = categories[i]
        keep = mask[i]
        if keep:
            filtered_categories.append(category)
    
    # now to encode categories before calculating score
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    true_labels = encoder.fit_transform(filtered_categories)
    
    # calculate score
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(true_labels, filtered_topics)
    nmi = normalized_mutual_info_score(true_labels, filtered_topics)
    
    # clustering success rate
    clustering_success_rate = clustered / len(topics)
    
    return ari, nmi, clustering_success_rate