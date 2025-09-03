# all the imports from above
from evaluate_clustering import evaluate_clustering_detailed
from supervision import basic_supervision
from zero_shot import zero_shot
from bayesian import bayesian
from coherence import coherence
from iterative import iterative
from ensemble import ensemble
from enhanced_cluster import enhanced_representation
from kmeans import kmeans
import multiprocessing

# global variables
maxARI = -1
maxAriMethod = ""
nmiBest = -1
nmiBestMet = ""

def update_max(method, ari, nmi):
   global maxARI, maxAriMethod, nmiBest, nmiBestMet
   if ari > maxARI:
       maxARI = ari
       maxAriMethod = method
   if nmi > nmiBest:
       nmiBest = nmi
       nmiBestMet = method


# basic_supervision - needs sup boolean
model, topics, categories = basic_supervision(True)
ari, nmi, success = evaluate_clustering_detailed(topics, categories)
update_max("basic_supervision_sup", ari, nmi)
print(f"basic_supervision supervised: ARI={ari:.3f}, NMI={nmi:.3f}")

model, topics, categories = basic_supervision(False)
ari, nmi, success = evaluate_clustering_detailed(topics, categories)
update_max("basic_supervision_unsup", ari, nmi)
print(f"basic_supervision unsupervised: ARI={ari:.3f}, NMI={nmi:.3f}")


model, topics, categories = bayesian()
ari, nmi, success = evaluate_clustering_detailed(topics, categories)
update_max("bayesian", ari, nmi)
print(f"bayesian supervised: ARI={ari:.3f}, NMI={nmi:.3f}")

# enhanced_representation - needs sup boolean
model, topics, categories = enhanced_representation(True)
ari, nmi, success = evaluate_clustering_detailed(topics, categories)
update_max("enhanced_representation_sup", ari, nmi)
print(f"enhanced_representation supervised: ARI={ari:.3f}, NMI={nmi:.3f}")

model, topics, categories = enhanced_representation(False)
ari, nmi, success = evaluate_clustering_detailed(topics, categories)
update_max("enhanced_representation_unsup", ari, nmi)
print(f"enhanced_representation unsupervised: ARI={ari:.3f}, NMI={nmi:.3f}")

# iterative - needs sup boolean
model, topics, categories = iterative(True)
ari, nmi, success = evaluate_clustering_detailed(topics, categories)
update_max("iterative_sup", ari, nmi)
print(f"iterative supervised: ARI={ari:.3f}, NMI={nmi:.3f}")

model, topics, categories = iterative(False)
ari, nmi, success = evaluate_clustering_detailed(topics, categories)
update_max("iterative_unsup", ari, nmi)
print(f"iterative unsupervised: ARI={ari:.3f}, NMI={nmi:.3f}")

model, topics, categories = zero_shot()
ari, nmi, success = evaluate_clustering_detailed(topics, categories)
update_max("zero_shot", ari, nmi)
print(f"zero_shot: ARI={ari:.3f}, NMI={nmi:.3f}")

model, topics, categories = coherence()
ari, nmi, success = evaluate_clustering_detailed(topics, categories)
update_max("coherence", ari, nmi)
print(f"coherence: ARI={ari:.3f}, NMI={nmi:.3f}")

model, topics, categories = ensemble()
ari, nmi, success = evaluate_clustering_detailed(topics, categories)
update_max("ensemble", ari, nmi)
print(f"ensemble: ARI={ari:.3f}, NMI={nmi:.3f}")

model, topics, categories = kmeans()
ari, nmi, success = evaluate_clustering_detailed(topics, categories)
update_max("kmeans_clustering", ari, nmi)
print(f"kmeans_clustering: ARI={ari:.3f}, NMI={nmi:.3f}")

print(f"\nBest ARI: {maxARI:.3f} - {maxAriMethod}")
print(f"Best NMI: {nmiBest:.3f} - {nmiBestMet}")