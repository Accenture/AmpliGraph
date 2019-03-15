from ampligraph.datasets import load_wn18
from ampligraph.latent_features import ComplEx
from ampligraph.evaluation import evaluate_performance, hits_at_n_score, mrr_score


X = load_wn18()

model = ComplEx(batches_count = 10,
                seed= 0,
                epochs = 20,
                k = 50,
                eta = 2,
                loss = "nll",
                optimizer = "adam",
                optimizer_params = {"lr":0.01})

model.fit(X['train'])

y_pred = model.predict(X['test'][:5,])

from scipy.special import expit
print(expit(y_pred))

ranks = evaluate_performance(X['test'][:10], model=model)
print(ranks)

mrr = mrr_score(ranks)
hits_10 = hits_at_n_score(ranks, n=10)
print("MRR: %f, Hits@10: %f" % (mrr, hits_10))

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

embs = model.get_embeddings(embs_labels, type='entity')
embs_2d = TSNE(n_components=2).fit_transform(embs)

fig, ax = plt.subplots()
ax.scatter(embs_2d[:, 0], embs_2d[:, 1])
for i, lab in enumerate(embs_labels):
    ax.annotate(lab, (embs_2d[i, 0], embs_2d[i, 1]))

plt.show(fig)


