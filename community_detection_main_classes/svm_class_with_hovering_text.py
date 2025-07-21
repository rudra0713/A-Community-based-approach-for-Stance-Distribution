import numpy as np
import sys
sys.path.insert(1, '/scratch/rrs99/Stance_Distribution/')
import torch
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer
from similarity_score_codes.return_bert_sim_score_2 import compute_sim_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



class SVM_class:
    def __init__(self, sentences, number_of_controversial_args=4):
        self.sentences = sentences
        self.number_of_controversial_args = number_of_controversial_args

    # def compute_embedding(self, article):
    #     embedder = SentenceTransformer('stsb-mpnet-base-v2')
    #
    #     corpus_embeddings = embedder.encode(article, convert_to_tensor=True)
    #     return corpus_embeddings

    def embedding_reduction(self, train_embedding):
        tsne_components = 2
        do_pca = True
        tsne = TSNE(n_components=tsne_components, verbose=1, perplexity=len(train_embedding) - 1, n_iter=10000, metric='cosine',
                    square_distances=True, random_state=7, method='exact')
        # if do_pca:
        #     pca_50 = PCA(n_components=50 if len(X) > 50 else 20, random_state=7)
        #     pca_result_50 = pca_50.fit_transform(train_embedding)
        #     print('Cumulative explained variation for 50 principal components: {}'.format(
        #         np.sum(pca_50.explained_variance_ratio_)))
        #
        #     tsne_results = tsne.fit_transform(pca_result_50)
        # else:
        #     tsne_results = tsne.fit_transform(train_embedding)

        tsne_results = tsne.fit_transform(train_embedding)
        return tsne_results

    def build_svm(self):
        doc_list = [doc for _, doc, _ in self.sentences]
        print("doc list ... ")
        print(doc_list)
        # sent_topic_words_embedding = compute_sim_score(None, self.sentences, compute_max=False, return_embedding=True)
        train_embeddings = compute_sim_score(None, doc_list, compute_max=False, return_embedding=True)
        train_embeddings = self.embedding_reduction(train_embeddings)
        train_label = [stance_label for _, _, stance_label in self.sentences]
        if len(set(train_label)) == 1:
            print("uni-modal stance distribution")
        else:
            svm_model = SVC(kernel='rbf')
            if torch.cuda.is_available():
                # pushing tensor to cpu first
                train_embeddings = train_embeddings.cpu().numpy()

            svm_model.fit(train_embeddings, train_label)
            print("svm model trained")
            y_hat = svm_model.decision_function(train_embeddings)
            print(y_hat)
            y_hat = [abs(x) for x in y_hat]

            # return the indices of the k smallest elements
            res = sorted(range(len(y_hat)), key=lambda sub: y_hat[sub])[:self.number_of_controversial_args]
            print("most controversial arguments .. ")
            for ind in res:
                print(self.sentences[ind][0], '--', self.sentences[ind][1], '--', self.sentences[ind][2])
            print(".......")
            self.plot_svm(model=svm_model, train_data=train_embeddings, train_label=train_label, text_data=doc_list)

    def plot_svm(self, model, train_data, train_label, text_data):

        def make_meshgrid(x0, x1, h=.02):
            x0_min, x0_max = x0.min() - 1, x0.max() + 1
            x1_min, x1_max = x1.min() - 1, x1.max() + 1
            print("x0_min and x0_max: ", x0_min, x0_max)
            print("x0_min and x1_max: ", x1_min, x1_max)
            xx, yy = np.meshgrid(np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h))
            return xx, yy

        def plot_contours(ax, clf, xx, yy, **params):
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            out = ax.contourf(xx, yy, Z, **params)
            return out

        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            # print(ind)
            text = "{}, {}".format(" ".join(list(map(str, ind["ind"]))),
                                   " ".join([text_data[n] for n in ind["ind"]]))
            annot.set_text(text)
            annot.get_bbox_patch().set_facecolor(cmap(norm(train_label[ind["ind"][0]])))
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        norm = plt.Normalize(1, 4)

        fig, ax = plt.subplots()
        # title for the plots
        title = ('Decision surface of linear SVC ')
        # Set-up grid for plotting.
        X0, X1 = train_data[:, 0], train_data[:, 1]
        xx, yy = make_meshgrid(X0, X1)
        cmap = plt.cm.coolwarm
        plot_contours(ax, model, xx, yy, cmap=cmap, alpha=0.8)

        sc = ax.scatter(X0, X1, c=train_label, cmap=cmap, s=20, edgecolors='k')
        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        ax.set_ylabel('y label here')
        ax.set_xlabel('x label here')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.legend()

        fig.canvas.mpl_connect("motion_notify_event", hover)
        # plt.show()
        plt.savefig('community-sample.jpg')


if __name__ == '__main__':
    inp = [('A', 'I support abortion', 0), ('B', 'Abortion should be legalized immediately', 0), ('C', 'I am against abortion', 1), ('D', 'Abortion has been banned by the supreme court and it is the right decision.', 1)]
    s = SVM_class(sentences=inp)
    s.build_svm()
    pass
    # # X = np.array([[-1, -1, 2], [-2, -1, 3], [1, 1, -1], [2, 1, 0.56], [3, 4, 0.45], [4, 0.233, 3], [0, 1, 2], [2, 0, 3]])
    # X = np.array([[1, 1], [2, 2], [2, 4], [4, 1], [-3, -4], [-4, -0.233], [-2, -1], [-2, -5]])
    # y = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    # clf = SVC(kernel='linear')
    # clf.fit(X, y)
    # y_hat = clf.decision_function(X)
    # print(y_hat)
    # w_norm = np.linalg.norm(clf.coef_)
    # dist = y_hat / w_norm
    # dist = [abs(x) for x in dist]
    # print(dist)
    # res = sorted(range(len(dist)), key=lambda sub: dist[sub])[:4]
    # print(res)
    # print("....")
    # clf = SVC(kernel='rbf')
    # clf.fit(X, y)
    # y_hat = clf.decision_function(X)
    # print(y_hat)
    # y_hat = [abs(x) for x in y_hat]
    #
    # res = sorted(range(len(y_hat)), key=lambda sub: y_hat[sub])[:4]
    # print(res)




