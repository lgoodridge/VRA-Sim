import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

"""
using training data from MovieLens dataset
citing: http://blog.ethanrosenthal.com/2015/11/02/intro-to-collaborative-filtering/
"""


def import_data():
    col = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv('ml-100k/u.data', sep='\t', names=col)
    data.head()
    return data

def setup_data(df):
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    #print str(n_users) + ' users'
    #print str(n_items) + ' items'
    
    ratings = np.zeros((n_users, n_items))
    for row in df.itertuples():
        ratings[row[1]-1, row[2]-1] = row[3]
    #print ratings
    
    sparsity = float(len(ratings.nonzero()[0]))
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    #print 'Sparsity: {:4.2f}%'.format(sparsity)
    return ratings

def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in xrange(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=10, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test

def calc_similarity(ratings, kind='user', epsilon=1e-9):
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

def predict_similarity(ratings, similarity, kind='user'):
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

def calc_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

def predict_topk(ratings, similarity, kind='user', k=40):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in xrange(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
            for j in xrange(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
    if kind == 'item':
        for j in xrange(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
            for i in xrange(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))        
    
    return pred

def main():
    data = import_data()
    #print data.head()
    ratings = setup_data(data)
    train, test = train_test_split(ratings)
    
    user_similarity = calc_similarity(train, kind='user')
    item_similarity = calc_similarity(train, kind='item')

    item_prediction = predict_similarity(train, item_similarity, kind='item')
    user_prediction = predict_similarity(train, user_similarity, kind='user')

    print 'User-based CF MSE: ' + str(calc_mse(user_prediction, test))
    print 'Item-based CF MSE: ' + str(calc_mse(item_prediction, test))

    pred = predict_topk(train, user_similarity, kind='user', k=40)
    print 'Top-k User-based CF MSE: ' + str(calc_mse(pred, test))

    pred = predict_topk(train, item_similarity, kind='item', k=40)
    print 'Top-k Item-based CF MSE: ' + str(calc_mse(pred, test))

    
if __name__ == "__main__":
  main()
