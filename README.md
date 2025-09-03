# Clustering Project

This is a collection of methods used to cluster the [Top 100 AI/ML interview questions and answers](https://razorops.com/blog/top-100-ai-ml-interview-questions-and-answers) based on the provided ground truth labels. 

These methods include but are not limited to:

- Hyperparameter Bayesian optimization
- Coherence-based clustering
- Aggressive, Balanced, and conservative hyperparameters
- Enhanced representation with KeyBert (and larger embeddings)
- Iterative improvements (by reducing outliers)
- KMeans clustering
- Zero Shot Clustering (provided ground truth categories)

Each model is evaluated based on the Rand index (ARI) and normalized mutual information (NMI) metrics. 

Clustering is done in the following manner:
```
raw text --> text preprocessing --> sentence embeddings --> UMAP --> HDBSCAN clustering --> topics
```
Preprocessing of the raw text was done by copy-pasting the questions, then processing them with a custom script (./src/utilities). The data was then saved as a .csv file in ./src/data.

## To Run:

First, clone the repository:

```bash
git clone https://github.com/milhud/ClusteringProject.git
cd ClusteringProject
```

Then, install the requirements:
```bash
pip install -r requirements.txt
```

Finally, run the main python file:
```bash
cd src
python main.py
```