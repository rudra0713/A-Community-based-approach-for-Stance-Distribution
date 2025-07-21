import numpy as np
import sys
sys.path.insert(1, '/scratch/rrs99/Stance_Distribution/')
import torch, matplotlib, os
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer
from similarity_score_codes.return_bert_sim_score_2 import compute_sim_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2
from datetime import datetime


class SVM_class:
    def __init__(self, claim, sentences, community_index, svm_dir_path, number_of_controversial_args=4, points_to_draw=2):
        self.sentences = sentences
        self.claim = claim
        self.community_index = community_index
        self.number_of_controversial_args = number_of_controversial_args
        self.train_embeddings = None
        self.train_label = None
        self.point_to_draw = points_to_draw
        self.svm_dir_path = svm_dir_path
        if not os.path.exists(self.svm_dir_path):
            os.makedirs(self.svm_dir_path)


    def embedding_reduction(self, train_embedding):
        tsne_components = 2
        do_pca = True
        tsne = TSNE(n_components=tsne_components, verbose=1, perplexity=len(train_embedding) - 1, n_iter=10000,
                    metric='cosine',  random_state=7, method='exact')
        # if do_pca:
        #     pca_50 = PCA(n_components=50 if len(X) > 50 else 20, random_state=7)
        #     pca_result_50 = pca_50.fit_transform(train_embedding)
        #     # # print('Cumulative explained variation for 50 principal components: {}'.format(
        #         np.sum(pca_50.explained_variance_ratio_)))
        #
        #     tsne_results = tsne.fit_transform(pca_result_50)
        # else:
        #     tsne_results = tsne.fit_transform(train_embedding)

        tsne_results = tsne.fit_transform(train_embedding)
        return tsne_results

    def create_embeddings(self):
        doc_list = [doc for _, doc, _ in self.sentences]
        # # print("doc list ... ")
        # print(doc_list)
        # sent_topic_words_embedding = compute_sim_score(None, self.sentences, compute_max=False, return_embedding=True)
        self.train_embeddings = compute_sim_score(None, doc_list, compute_max=False, return_embedding=True)
        self.train_label = [stance_label for _, _, stance_label in self.sentences]
        if torch.cuda.is_available():
            # pushing tensor to cpu first
            self.train_embeddings = self.train_embeddings.cpu().numpy()
        self.train_embeddings = self.embedding_reduction(self.train_embeddings)

    def build_svm(self, num_points):
        current_train_embeddings = self.train_embeddings[:num_points]
        current_train_labels = self.train_label[:num_points]

        svm_model = SVC(kernel='rbf')
        svm_model.fit(current_train_embeddings, current_train_labels)

        return svm_model

    def build_svm_figures(self):
        self.create_contour_video()
        self.plot_svm()

    def make_meshgrid(self, x0, x1, h=5):
        x0_min, x0_max = x0.min() - 10, x0.max() + 10
        x1_min, x1_max = x1.min() - 10, x1.max() + 10
        # print("x0_min and x0_max: ", x0_min, x0_max)
        # print("x1_min and x1_max: ", x1_min, x1_max)
        xx, yy = np.meshgrid(np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h))
        return xx, yy

    def plot_contours(self, ax, clf, xx, yy, **params):
        temp = np.c_[xx.ravel(), yy.ravel()]
        Z = clf.predict(temp)
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    def create_contour_video(self):
        self.create_embeddings()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.svm_dir_path + os.sep + 'output-community-' + str(self.community_index) + '.mp4', fourcc, 20.0, (640, 480))

        # Set up the scatter plot
        fig, ax = plt.subplots()
        annotations = []
        point_counter = self.point_to_draw
        X0, X1 = self.train_embeddings[:, 0], self.train_embeddings[:, 1]
        ax.set_xlim([X0.min() - 10, X0.max() + 10])
        ax.set_ylim([X1.min() - 10, X1.max() + 10])
        sc = ax.scatter([], [], s=40, facecolor='None')
        contour = None
        # sc = ax.scatter([], [], s=40)

        # Generate some initial data
        # x = np.random.randint(0, 10, size=5)
        # y = np.random.randint(0, 10, size=5)
        # labels = np.random.randint(0, 2, size=5)
        # cmap = plt.cm.coolwarm
        # xx, yy = self.make_meshgrid(X0, X1)
        # # print("going to plot contours for community: ", self.community_index)
        # self.plot_contours(ax, model, xx, yy, cmap=cmap, alpha=0.2)

        sc.set_offsets(self.train_embeddings[:point_counter])
        sc.set_array(self.train_label[:point_counter])
        # sc.set_facecolor(['white' if label == 0 else 'white' for label in train_label[:point_counter]])
        sc.set_facecolor('None')
        sc.set_edgecolor(['green' if label == 0 else 'red' for label in self.train_label[:point_counter]])

        for i in range(min(point_counter, len(self.sentences))):
            annotations.append(ax.annotate(self.sentences[i][0], xy=(X0[i], X1[i]), xytext=(0, 10), textcoords="offset points", color='black'))
            # annotations.append(ax.annotate(str(i), (x[i], y[i]), color='white'))
        last_drawn_time = 0
        # Loop to add more points to the scatter plot and create the video
        # loop for 300 is equivalent to 15 second
        loop_counter = (len(self.sentences) // point_counter + 1) * 50
        for i in range(loop_counter):
            # Add new data every 5 seconds
            if i % 50 == 0 and i > 0:
                point_counter += self.point_to_draw
                last_drawn_time = i
                # new_x = np.random.randint(0, 10, size=5)
                # new_y = np.random.randint(0, 10, size=5)
                # new_labels = np.random.randint(0, 2, size=5)
                # x = np.hstack((x, new_x))
                # y = np.hstack((y, new_y))
                # labels = np.hstack((labels, new_labels))
                # sc.set_offsets(np.vstack((x, y)).T)
                # sc.set_array(labels)
                # print("train label: ", self.train_label[:point_counter])
                sc.set_offsets(self.train_embeddings[:point_counter])
                sc.set_array(self.train_label[:point_counter])
                sc.set_facecolor('None')
                sc.set_edgecolor(['green' if label == 0 else 'red' for label in self.train_label[:point_counter]])

                for j in range(min(point_counter, len(self.sentences))):
                    annotations.append(ax.annotate(self.sentences[j][0], xy=(X0[j], X1[j]), xytext=(0, 10),
                                                   textcoords="offset points", color='black'))

                # for j in range(len(new_x)):
                #     annotations.append(ax.annotate(str(len(x) - len(new_x) + j), (new_x[j], new_y[j]), color='white'))
            if i == last_drawn_time + 10:
                current_X0, current_X1 = self.train_embeddings[:point_counter, 0], self.train_embeddings[:point_counter, 1]
                if len(set(self.train_label[:point_counter])) > 1:
                    model = self.build_svm(point_counter)
                    # xx, yy = self.make_meshgrid(current_X0, current_X1)
                    xx, yy = self.make_meshgrid(X0, X1)

                    # # print(xx, yy)
                    cmap = plt.cm.coolwarm
                    # print("going to plot contours for community: ", self.community_index)
                    if contour:
                        # print("cleared contour ... ")
                        for c in contour.collections:
                            c.remove()
                    # print("drawing contour for {} points".format(point_counter))
                    contour = self.plot_contours(ax, model, xx, yy, cmap=cmap, alpha=0.2)
                else:
                    pass
                    # print("unimodal distribution ... ")

            # Update the plot and write to video
            # # print("point counter: ", point_counter)
            # if point_counter > len(self.sentences):
            #     xx, yy = self.make_meshgrid(X0, X1)
            #     # # print(xx, yy)
            #     cmap = plt.cm.coolwarm
            #     # print("going to plot contours for community: ", self.community_index)
            #     self.plot_contours(ax, model, xx, yy, cmap=cmap, alpha=0.2)

            fig.canvas.draw()
            frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            for annotation in annotations:
                annotation.remove()
            annotations = []
            for j in range(min(point_counter, len(self.sentences))):
                annotations.append(
                    ax.annotate(self.sentences[j][0], xy=(X0[j], X1[j]), xytext=(0, 10), textcoords="offset points", color='black'))

            sc.set_offsets(self.train_embeddings[:point_counter])
            sc.set_array(self.train_label[:point_counter])
            sc.set_facecolor('None')
            sc.set_edgecolor(['green' if label == 0 else 'red' for label in self.train_label[:point_counter]])

            # for j in range(len(x)):
            #     annotations.append(ax.annotate(str(j), (x[j], y[j]), color='white'))
            # sc.set_offsets(train_data[:point_counter])
            # sc.set_array(train_label[:point_counter])
            # sc.set_facecolor(['white' if label == 0 else 'white' for label in train_label[:point_counter]])
            # sc.set_edgecolor(['green' if label == 0 else 'red' for label in train_label[:point_counter]])

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)

        out.release()
        plt.close()

    def plot_svm(self):

        fig, ax = plt.subplots()
        # title for the plots
        title = ''
        # Set-up grid for plotting.
        X0, X1 = self.train_embeddings[:, 0], self.train_embeddings[:, 1]
        xx, yy = self.make_meshgrid(X0, X1)
        # # print(xx, yy)
        cmap = plt.cm.copper
        colors = ['green', 'red']
        if len(set(self.train_label[:len(self.sentences)])) > 1:
            print("going to plot contours for community: ", self.community_index)
            model = self.build_svm(num_points=len(self.sentences))  # building model on full data
            self.plot_contours(ax, model, xx, yy, cmap=cmap, alpha=0.8)

            # print("going to generate scatter plot for community: ", self.community_index)
            sc = ax.scatter(X0, X1, c=self.train_label, cmap=matplotlib.colors.ListedColormap(colors), s=40, edgecolors='k')

            # print("going to annotate for community: ", self.community_index)
            for i, (index, arg, stance) in enumerate(self.sentences):
                ax.annotate(index, xy=(X0[i], X1[i]), xytext=(0, 10), textcoords="offset points")

            # annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
            #                     bbox=dict(boxstyle="round", fc="w"),
            #                     arrowprops=dict(arrowstyle="->"))
            # annot.set_visible(False)

            ax.set_ylabel('coordinate_y')
            ax.set_xlabel('coordinate_x')
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(title)
            # ax.legend()

            # fig.canvas.mpl_connect("motion_notify_event", hover)
            # plt.show()
            # print("going to save image for community: ", self.community_index)
            plt.savefig(self.svm_dir_path + os.sep + 'community-' + str(self.community_index) + '.jpg')
            output_file_name = 'all_communities.txt'
            output_file_path = self.svm_dir_path + os.sep + output_file_name
            # print("trying to open file")
            com_file = open(output_file_path, 'a+')

            com_file.write('community-' + str(self.community_index) + '\n')
            for ind in range(len(self.sentences)):
                com_file.write(self.sentences[ind][0] + '--' + self.sentences[ind][1] + '--' + str(self.sentences[ind][2]) + '\n')
            com_file.write('\n')
            com_file.close()


if __name__ == '__main__':
    inp = [('A', 'I support abortion', 0), ('B', 'I am against abortion', 1), ('C', 'Abortion should be legalized immediately', 0), ('D', 'Abortion has been banned by the supreme court and it is the right decision.', 1),
           ('E', 'Adoption is a better choice', 1), ('F', 'It is the mother\'s choice', 0),
           ('M', 'Abortion should be banned right now.', 1), ('N', 'Abortion is murder.', 1),
           ('O', 'Government cannot dictate individual choice', 0), ('P', 'Dont force your opinion on us.', 0)
           ]
    s = SVM_class(claim='Abortion should be legalized.', sentences=inp, community_index=1)
    # s.create_contour_video()
    s.build_svm_figures()

    inp_2 = [('G', 'US supreme court supports abortion.', 0), ('H', 'Abortion should be valid', 0), ('I', 'Abortion is cruel.', 1), ('J', 'Abortion is basically murder.', 1),
           ('K', 'Abortion is immoral.', 1), ('L', 'It is the mother\'s choice and mother\'s choice only', 0)]
    s2 = SVM_class(claim='Abortion should be legalized.', sentences=inp_2, community_index=2)
    # s2.create_contour_video()
    s2.build_svm_figures()
    pass
    # # X = np.array([[-1, -1, 2], [-2, -1, 3], [1, 1, -1], [2, 1, 0.56], [3, 4, 0.45], [4, 0.233, 3], [0, 1, 2], [2, 0, 3]])
    # X = np.array([[1, 1], [2, 2], [2, 4], [4, 1], [-3, -4], [-4, -0.233], [-2, -1], [-2, -5]])
    # y = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    # clf = SVC(kernel='linear')
    # clf.fit(X, y)
    # y_hat = clf.decision_function(X)
    # # print(y_hat)
    # w_norm = np.linalg.norm(clf.coef_)
    # dist = y_hat / w_norm
    # dist = [abs(x) for x in dist]
    # # print(dist)
    # res = sorted(range(len(dist)), key=lambda sub: dist[sub])[:4]
    # # print(res)
    # # print("....")
    # clf = SVC(kernel='rbf')
    # clf.fit(X, y)
    # y_hat = clf.decision_function(X)
    # # print(y_hat)
    # y_hat = [abs(x) for x in y_hat]
    #
    # res = sorted(range(len(y_hat)), key=lambda sub: y_hat[sub])[:4]
    # # print(res)




