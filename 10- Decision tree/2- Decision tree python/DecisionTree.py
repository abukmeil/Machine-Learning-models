import numpy as np
from collections import Counter



class Node:
    """Python class to retuen oject at each node and the leaf node
    """
    def __init__(self,best_feature_idx=None,best_threshold=None,left_node=None,right_node=None,*,label_value=None):    
        self.best_feature_idx=best_feature_idx
        self.best_threshold=best_threshold
        self.left_node=left_node
        self.right_node=right_node
        self.label_value=label_value
   
    def is_leaf(self):
        return self.label_value is not None
        
        
class DecisionTree:
    """ Python class for decison tree model

    Returns:
        Python class object: Node the hold the best label if it is a leaf node, elase another node with other information
    """
    # identifying the min sample for splitting and the # of feature points in the  feature vector for training
    def __init__(self,min_samples_for_splitting=2,max_depth=100,n_feats=None): # n_feats can be randomly sampled
        self.min_samples_for_splitting=min_samples_for_splitting
        self.max_depth=max_depth
        self.n_feats=n_feats
        self.root_node=None
    
    def fit(self,X,y):
        self.n_feats=X.shape[1] if not self.n_feats else min(self.n_feats,X.shap1)
        
        self.root_node=self._grow_tree(X,y)
    def _grow_tree(self,X,y,depth=0):

        n_samples,n_features=X.shape
        n_labels= len(np.unique(y))
        
        # Stopping criteria
        if (n_labels==1
            or n_samples <self.min_samples_for_splitting
            or depth>self.max_depth):
            leaf_node=self._most_common_label(y)
            return Node(label_value=leaf_node)
        
        feat_idxs=np.random.choice(n_features,self.n_feats,replace=False)
        best_feature_idx,best_threshold=self._best_criteria(X,y,feat_idxs)
        left_node_idxs,right_node_idxs=self._split_root_node(X[:,best_feature_idx],best_threshold)
        left_node=self._grow_tree(X[left_node_idxs,:],y[left_node_idxs],depth+1)
        right_node=self._grow_tree(X[right_node_idxs,:],y,depth+1)
        return Node(best_feature_idx=best_feature_idx,best_threshold=best_threshold,left_node=left_node,right_node=right_node)
        
        
    def _best_criteria(self,X,y,feat_idxs):
        best_gain=-1
        best_feature_idx,best_threshold=None,None
        for idx in feat_idxs:
            X_c=X[:,idx]
            thresholds=np.unique(X_c)
            for threshold in thresholds:
                info_gain=self._information_gain(X_c,y,threshold)
                if info_gain>best_gain:
                    best_gain=info_gain
                    best_feature_idx=idx
                    best_threshold=threshold
        return best_feature_idx,best_threshold
    
    def _information_gain(self,X_c,y,threshold):
        
        root_node_enropy=self._entropy(y)
        
        # entropy for left, right nodes
        left_node_idxs, right_node_idxs=self._split_root_node(X_c,threshold)
        entropy_left,entropy_right=self._entropy(y[left_node_idxs]), self._entropy(y[right_node_idxs])
        
        # Child entropy
        l_y=len(y)
        l_l=len(left_node_idxs)
        l_r=len(right_node_idxs)
        
        child_entropy= (l_l/l_y)*entropy_left + (l_r/l_y)*entropy_right
        info_gain=root_node_enropy-child_entropy
        
        return info_gain
        
        
    def _split_root_node(self, X_c,threshold):
        left_node_idxs=np.argwhere(X_c <= threshold).flatten()
        right_node_idxs=np.argwhere(X_c>threshold).flatten()
        return left_node_idxs,right_node_idxs
    
    
    def _entropy(self,y):
        hist=np.bincount(y)
        ps=hist/len(y)
        return - np.sum([p* np.log2(p) for p in ps if p >0 ])
    
         
    def _most_common_label(self,y):
        counter=Counter(y)
        most_common_label=counter.most_common(1)[0][0]
        return most_common_label
        
    def predict(self,X):
        return np.array([self._traverse_tree(x,self.root_node) for x in X])
    
    def _traverse_tree(self,x,node):
        if node.is_leaf():
            return node.label_value
        if x[node.best_feature_idx] <=node.best_threshold:
            return self._traverse_tree(x,node.left_node)
        return self._traverse_tree(x,node.right_node)
        
    def model_accuracy(self,y_true,y_predicted):
        return np.sum(y_true==y_predicted)/len(y_true)
   
        
            
        
        