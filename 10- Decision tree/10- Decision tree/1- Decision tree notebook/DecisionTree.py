import numpy as np
from collections import Counter

class Node:
    def __init__(self,best_feature_idx=None,best_threshold=None,left_node=None,right_node=None,*,label_value=None):    
        self.best_feature_idx=best_feature_idx
        self.best_threshold=best_threshold
        self.Left_node=left_node
        self.right_node=right_node 
        self.label_value=label_value # This is a keyword argument because the node is a leaf we return the Node object by passing this value
        
        # Check if the node is a leaf node or not
    def is_leaf(self):
        return self.label_value is not None # Not a leaf value if no value
        
class DecisionTree:
    # identifying the min sample for splitting and the # of feature points in teh feature vector for training
    def __init__(self,min_samples_for_splitting=2,max_depth=100,n_feats=None): # n_feats can be randomly sampled
        self.min_samples_for_splitting=min_samples_for_splitting
        self.max_depth=max_depth
        self.n_feats=n_feats
        self.root_node=None
            
    def fit(self,X,y):
        self.n_feats=X.shape[1] if not self.n_feats else min(self.n_feats,X.shape[1])
        self.root_node=self._grow_tree(X,y) # at the end root_node will be a python class object
        
    def _grow_tree(self,X,y,depth=0):
        n_samples,n_features=X.shape
        n_labels=len(np.unique(y))
        
        # Before growing check the stopping criteria for recursion
        if (n_labels==1
            or n_samples<self.min_samples_for_splitting
            or depth>= self.max_depth):
            leaf_value=self._most_common_label(y)
            return Node(label_value=leaf_value)
        feat_idxs=np.random.choice(n_features,self.n_feats,replace=False) # True to avoid picking the same index twice
        best_feature_idx,best_threshold=self._best_criteria(X,y,feat_idxs)
        left_idxs,right_idxs=self._split(X[:,best_feature_idx],best_threshold)
        left_node=self._grow_tree(X[left_idxs,:],y[left_idxs],depth+1)
        right_node=self._grow_tree(X[right_idxs,:],y[right_idxs],depth+1)
        return Node(best_feature_idx=best_feature_idx,best_threshold=best_threshold,left_node=left_node,right_node=right_node)
            # We do not add the leaf value at this node object till reaching the leaf node (see above)
        
        
    def _best_criteria(self,X,y,feat_idxs):
        best_info_gain=-1
        best_feature_idx,best_threshold=None,None
        for idx in feat_idxs:
            X_c=X[:,idx]
            thresholds=np.unique(X_c)
            for threshold in thresholds:
                info_gain=self._information_gain(y,X_c,threshold)
                if info_gain > best_info_gain:
                    best_info_gain=info_gain
                    best_feature_idx=idx
                    best_threshold=threshold
        return best_feature_idx,best_threshold
    
    def _information_gain(self,y,X_c,threshold):
        root_node_entropy=self._entropy(y)
        # left and right nodes entropy
        left_idxs,right_idxs=self._split(X_c,threshold)
        len_left=len(left_idxs)
        len_right=len(right_idxs)
        if len_left==0 or len_right==0:
            return 0
        
        entropy_left=self._entropy(y[left_idxs])
        entropy_right=self._entropy(y[right_idxs])
        # child weighted  entropy
        child_entropy= (len_left/len(y))*entropy_left +(len_right/len(y))*entropy_right
        
        info_gain=root_node_entropy-child_entropy
        return info_gain
        
    def _split(self,X_c,threshold):
        left_idxs=np.argwhere(X_c<=threshold).flatten()
        right_idxs=np.argwhere(X_c>threshold).flatten()
        return left_idxs ,right_idxs
        
    def _entropy(self,y):
        hist=np.bincount(y)
        ps=hist/len(y)
        return - np.sum([p*np.log2(p) for p in ps if p > 0]) # the log is indefined for 0
        
    def _most_common_label(self,y):
        counter=Counter(y)
        most_common=counter.most_common(1)[0][0] # Slicing the first value ofa list of tuple


    def predict(self,X):
        return np.array([self._traverse_tree(x,self.root_node) for x in X])
    def _traverse_tree(self,x,node):
        if node.is_leaf():
            return node.label_value    
        if x[node.best_feature_idx] <= node.best_threshold:
            return self._traverse_tree(x,node.left_node)
        return self._traverse_tree(x,node.right_node)
        

