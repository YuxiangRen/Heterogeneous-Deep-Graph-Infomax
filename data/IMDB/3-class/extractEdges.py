import pickle 
from collections import Counter
import scipy.sparse as sp
import numpy as np 

original_file = './movie_metadata_3class.csv'

movie_idx_map = {}
actor_idx_map = {}
director_idx_map = {}
keyword_idx_map = {}
with open('movie_idx_map.pickle', 'rb') as m:
        movie_idx_map = pickle.load(m)
with open('actor_idx_map.pickle', 'rb') as a:
        actor_idx_map = pickle.load(a)
with open('director_idx_map.pickle', 'rb') as d:
        director_idx_map = pickle.load(d)
with open('keyword_idx_map.pickle', 'rb') as k:
        keyword_idx_map = pickle.load(k)

movie_actor_edges = []
movie_director_edges = []
movie_keyword_edges = []
    
with open(original_file, 'r') as f:
    next(f)
    lines = f.readlines()
    for line in lines:
         line = line.split(',')
         movie = line[11]
         actor_1 = line[6]
         actor_2 = line[10]
         actor_3 = line[14]
         director = line[1]
         keywords = line[16].split('|')
         if [movie_idx_map[movie],actor_idx_map[actor_1]] not in movie_actor_edges:
             movie_actor_edges.append([movie_idx_map[movie],actor_idx_map[actor_1]])
         if [movie_idx_map[movie],actor_idx_map[actor_2]] not in movie_actor_edges:
             movie_actor_edges.append([movie_idx_map[movie],actor_idx_map[actor_2]])
         if [movie_idx_map[movie],actor_idx_map[actor_3]] not in movie_actor_edges:
             movie_actor_edges.append([movie_idx_map[movie],actor_idx_map[actor_3]])
         if [movie_idx_map[movie],director_idx_map[director]] not in movie_director_edges:
             movie_director_edges.append([movie_idx_map[movie],director_idx_map[director]])
         for keyword in keywords:
                 keyword_idx = keyword_idx_map[keyword]
                 if [movie_idx_map[movie],keyword_idx] not in movie_keyword_edges:
                     movie_keyword_edges.append([movie_idx_map[movie],keyword_idx])
                 
with open('movie_actor_edges.pickle', 'wb') as m:
        pickle.dump(movie_actor_edges, m)
m.close
with open('movie_director_edges.pickle', 'wb') as a:
        pickle.dump(movie_director_edges, a)
a.close
with open('movie_keyword_edges.pickle', 'wb') as d:
        pickle.dump(movie_keyword_edges, d)
d.close
movie_actor_edges = np.array(movie_actor_edges)
movie_actor_adj = sp.coo_matrix((np.ones(movie_actor_edges.shape[0]), (movie_actor_edges[:, 0], movie_actor_edges[:, 1])), shape=(len(movie_idx_map), len(actor_idx_map)), dtype=np.int32)
#movie_actor_adj =movie_actor_adj.todense()
movie_director_edges = np.array(movie_director_edges)
movie_director_adj = sp.coo_matrix((np.ones(movie_director_edges.shape[0]), (movie_director_edges[:, 0], movie_director_edges[:, 1])), shape=(len(movie_idx_map), len(director_idx_map)), dtype=np.int32)
#movie_director_adj =movie_director_adj.todense()
movie_keyword_edges = np.array(movie_keyword_edges)
movie_keyword_adj = sp.coo_matrix((np.ones(movie_keyword_edges.shape[0]), (movie_keyword_edges[:, 0], movie_keyword_edges[:, 1])), shape=(len(movie_idx_map), len(keyword_idx_map)), dtype=np.int32)
#movie_keyword_adj =movie_keyword_adj.todense()

with open('movie_actor_adj.pickle', 'wb') as m:
        pickle.dump(movie_actor_adj, m)
m.close
with open('movie_director_adj.pickle', 'wb') as a:
        pickle.dump(movie_director_adj, a)
a.close
with open('movie_keyword_adj.pickle', 'wb') as d:
        pickle.dump(movie_keyword_adj, d)
d.close


movie_actor_movie_adj = sp.coo_matrix.dot(movie_actor_adj,movie_actor_adj.transpose())
#movie_actor_movie_adj = movie_actor_movie_adj.todense()
movie_director_movie_adj = sp.coo_matrix.dot(movie_director_adj,movie_director_adj.transpose())
#movie_director_movie_adj = movie_director_movie_adj.todense()
movie_keyword_movie_adj = sp.coo_matrix.dot(movie_keyword_adj,movie_keyword_adj.transpose())
#movie_keyword_movie_adj = movie_keyword_movie_adj.todense()

matrix_temp = np.ones(movie_actor_movie_adj.shape)-np.eye(movie_actor_movie_adj.shape[0])
movie_actor_movie_adj = movie_actor_movie_adj.multiply(matrix_temp)
#movie_actor_movie_adj = movie_actor_movie_adj.todense()
movie_director_movie_adj = movie_director_movie_adj.multiply(matrix_temp)
#movie_director_movie_adj = movie_director_movie_adj.todense()
movie_keyword_movie_adj = movie_keyword_movie_adj.multiply(matrix_temp)
#movie_keyword_movie_adj = movie_keyword_movie_adj.todense()
homo_movie_adj = movie_actor_movie_adj + movie_director_movie_adj + movie_keyword_movie_adj
#homo_movie_adj = homo_movie_adj.todense()
with open('movie_actor_movie_adj.pickle', 'wb') as m:
        pickle.dump(movie_actor_movie_adj, m)
m.close
with open('movie_director_movie_adj.pickle', 'wb') as a:
        pickle.dump(movie_director_movie_adj, a)
a.close
with open('movie_keyword_movie_adj.pickle', 'wb') as d:
        pickle.dump(movie_keyword_movie_adj, d)
d.close
with open('homo_movie_adj.pickle', 'wb') as h:
        pickle.dump(homo_movie_adj, h)
h.close