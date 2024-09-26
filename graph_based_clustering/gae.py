
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.layers import GINConv, GlobalAvgPool, GCNConv

import scipy.sparse as sp
import numpy as np
from utils import * 



class GAE(object):
    def __init__(self, N, F, X, A, A_label, pos_weight, graph_a, val_edges, val_edges_false):
        self.N = N  # Number of nodes in the graph
        self.F = F  # Original size of node features
        self.X = X
        self.A = A
        self.A_label = A_label
        self.pos_weight = pos_weight
        self.graph_a = graph_a
        self.val_edges = val_edges
        self.val_edges_false = val_edges_false

        # Parameters
        self.hidden_dim1, self.hidden_dim2 = 32, 16  # Units in the GCN layers
        self.dropout = 0.0  # Dropout rate
        self.l2_reg = 0e-5  # L2 regularization rate
        self.learning_rate = 1e-2  # Learning rate
        self.epochs = 120  # Max number of training epochs
        self.val_epochs = 30  # After how many epochs should check the validation set

        # GNN architecture
        x_in = Input(shape=(self.F,))
        a_in = Input((self.N,), sparse=True)

        gc_1 = GCNConv(
            self.hidden_dim1,
            activation="relu",
            kernel_regularizer=l2(self.l2_reg),
        )([x_in, a_in])
        gc_1 = Dropout(self.dropout)(gc_1)
        z = GCNConv(
            self.hidden_dim2,
            activation=None,
            kernel_regularizer=l2(self.l2_reg),
        )([gc_1, a_in])

        graph_z = GlobalAvgPool()(z)

        out = tf.matmul(z, tf.transpose(z))
        A_rec = tf.keras.layers.Activation('sigmoid')(out)
        out = tf.reshape(out, [-1])

        self.model = Model(inputs=[x_in, a_in], outputs=[out, A_rec, z, graph_z])
        self.optimizer = Adam(learning_rate=self.learning_rate)

        # Define training step
        # @tf.function

    def train_step(self):
        with tf.GradientTape() as tape:
            predictions, _, _, _ = self.model([self.X, self.A], training=True)
            # predictions = tf.cast(predictions, tf.float64)
            # print(predictions.shape[0])
            # print(predictions.dtype)
            # print(self.A_label.dtype)
            if predictions.shape[0] > 583512330:
                # print("yes")
                loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=predictions[:25000000],
                                                                               labels=self.A_label[:25000000],
                                                                               pos_weight=self.pos_weight))
            else:
                loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=predictions[:25000000],
                                                                               labels=self.A_label[:25000000],
                                                                               pos_weight=self.pos_weight))
            # loss = tf.cast(loss, tf.float64)
            loss += sum(self.model.losses)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    # Training
    def train(self):
        best_val_roc = 0
        for epoch in range(1, self.epochs):
            loss = self.train_step()
            # print(f"epoch: {epoch:d} -- loss: {loss:.3f}")

            # Check performance on Validation
            if epoch % self.val_epochs == 0:
                _, adj_rec, _, _ = self.model([self.X, self.A])
                adj_rec = adj_rec.numpy()
                val_roc, _ = get_roc_score(self.val_edges, self.val_edges_false, adj_rec)

                if val_roc <= best_val_roc:
                    break
                else:
                    best_val_roc = val_roc
                    acc = np.mean(np.round(adj_rec) == self.graph_a.toarray())
                # print(f"Val AUC: {best_val_roc*100:.1f}, Accuracy: {acc*100:.1f}")

        return self.model

    
    
def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # This function is taken from the official repository: https://github.com/tkipf/gae

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]

    # add condition to set num_test to be the max (1, int(np.floor(edges.shape[0] / 10.)))
    # For small graph with less than 10 edges, the num_test and num_val would be 0
    num_test = max(1, int(np.floor(edges.shape[0] / 10.)))
    num_val = max(1, int(np.floor(edges.shape[0] / 20.)))
    # print(edges_all.shape[0])
    # print(edges.shape[0])
    # print(num_test)
    # print(num_val)

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[len(all_edge_idx) - num_test:]

    # test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    # print(all_edge_idx)
    # print(test_edge_idx)
    # print(val_edge_idx)
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    #     assert ~ismember(test_edges_false, edges_all)
    #     assert ~ismember(val_edges_false, edges_all)
    #     assert ~ismember(val_edges, train_edges)
    #     assert ~ismember(test_edges, train_edges)
    #     assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def get_roc_score(edges_pos, edges_neg, adj_rec):
    preds = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def graph_embed(graph):
    # Node features
    X = graph.x
    # X =  X.astype('float64')

    # Target graph to reconstruct
    A_label = graph.a + sp.eye(graph.a.shape[0], dtype=np.float32)
    A_label = A_label.toarray().reshape([-1])

    # Remove edges randomly from training set and put them in the validation/test sets
    adj_train, _, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(graph.a)

    # speed up computation
    # adj_train = normalized_adjacency(adj_train, True)
    # graph_adj = normalized_adjacency(graph.a, True)

    # Normalize the adj matrix and convert it to sparse tensor
    # normalize the matrix as #\(\D^{-\frac{1}{2}}\A\D^{-\frac{1}{2}}\);
    # Normalization is important for GCN

    A = GCNConv.preprocess(adj_train)
    A = sp_matrix_to_sp_tensor(A)

    # Compute the class weights (necessary due to imbalanceness in the number of non-zero edges)
    pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()

    N = graph.n_nodes
    F = graph.n_node_features

    gae = GAE(N, F, X, A, A_label, pos_weight, graph.a, val_edges, val_edges_false)

    encoder = gae.train()

    # Testing
    _, adj_rec, node_emb, graph_emb = encoder([X, A])
    adj_rec = adj_rec.numpy()
    roc_score, ap_score = get_roc_score(test_edges, test_edges_false, adj_rec)
    print(f"AUC: {roc_score * 100:.1f}, AP: {ap_score * 100:.1f}")
    # test_acc = np.mean(np.round(adj_rec.ravel()) == A_label)
    # print(f"Test accuracy: {test_acc*100:.1f}")

    # i+=1

    return graph_emb


