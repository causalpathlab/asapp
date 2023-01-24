import numpy as np
from model import _rpqr
import logging
logger = logging.getLogger(__name__)


class Node:
    def __init__(self):
        self.parent = None
        self.indxs = None
        self.pos_child = None
        self.neg_child = None

class StepTree:
    def __init__(self,mat,rp_mat):
        self.root = None
        self.mat = mat
        self.rp_mat = rp_mat
        self.add_root()
    
    def add_root(self):
        root = Node()
        root.level = 0
        root.indxs = list(range(self.mat.shape[0]))
        self.root = root
    
    def add_node(self,cnode,min_leaf,max_depth):

        if cnode.level < max_depth: 
            
            current_mat = self.mat[cnode.indxs]
            current_rp_mat = self.rp_mat[cnode.level,:]
            rpvec = np.asarray(np.dot(current_mat,current_rp_mat))
            rpvec = rpvec[0]    
            pos = []
            neg = []
            for indx,val in enumerate(rpvec):
                if val>= 0 : pos.append(cnode.indxs[indx])
                else : neg.append(cnode.indxs[indx])
            
            if len(pos)>0:
                new_pos_node = Node()
                new_pos_node.indxs = pos
                new_pos_node.level = cnode.level + 1
                cnode.pos_child = new_pos_node
                if len(pos)>min_leaf:
                    self.add_node(new_pos_node,min_leaf,max_depth)

            if len(neg)>0:
                new_neg_node = Node()
                new_neg_node.indxs = neg
                new_neg_node.level = cnode.level + 1
                cnode.neg_child = new_neg_node 
                if len(neg)>min_leaf:
                    self.add_node(new_neg_node,min_leaf,max_depth)
                    
    def build_tree(self,min_leaf,max_depth):
        self.add_node(self.root,min_leaf,max_depth)

    def print_tree(self):
        queue = []
        queue.append(self.root.pos_child)
        queue.append(self.root.neg_child)
        while queue:
            current_node = queue.pop(0)
            print(current_node.level,len(current_node.indxs))
            if current_node.pos_child != None:
                queue.append(current_node.pos_child)
            if current_node.neg_child != None:
                queue.append(current_node.neg_child)

    def make_bulk(self):
        leafd = {}
        queue = []
        if self.root.pos_child != None:
            queue.append(self.root.pos_child)
        if self.root.neg_child != None:
            queue.append(self.root.neg_child)
        i = 0
        while queue:
            current_node = queue.pop(0)
            if current_node.pos_child == None and current_node.neg_child == None: 
                leafd[i] = current_node.indxs
                i +=1
            else:            
                if current_node.neg_child != None:
                    queue.append(current_node.neg_child)
                if current_node.pos_child != None:
                    queue.append(current_node.pos_child) 
        return leafd  

class DCStepTree(StepTree):
    def __init__(self,mat,rp_mat,dc_mat):
        self.root = None
        self.mat = mat
        self.rp_mat = rp_mat
        self.dc_mat = dc_mat
        self.add_root()
    
    def add_node(self,cnode,min_leaf,max_depth):

        if cnode.level < max_depth: 
            
            current_rp_mat = self.rp_mat[cnode.level,:][:,np.newaxis]

            current_mat = self.mat[cnode.indxs]
            rpvec = np.asarray(np.dot(current_mat,current_rp_mat))

            current_dcmat = self.dc_mat[cnode.indxs]
            dcvec = np.asarray(np.dot(current_dcmat,current_rp_mat))

            rpdcvec = rpvec - dcvec
            
            pos = []
            neg = []
            for indx,val in enumerate(rpdcvec.flatten()):
                if val>= 0 : pos.append(cnode.indxs[indx])
                else : neg.append(cnode.indxs[indx])
            
            if len(pos)>0:
                new_pos_node = Node()
                new_pos_node.indxs = pos
                new_pos_node.level = cnode.level + 1
                cnode.pos_child = new_pos_node
                if len(pos)>min_leaf:
                    self.add_node(new_pos_node,min_leaf,max_depth)

            if len(neg)>0:
                new_neg_node = Node()
                new_neg_node.indxs = neg
                new_neg_node.level = cnode.level + 1
                cnode.neg_child = new_neg_node 
                if len(neg)>min_leaf:
                    self.add_node(new_neg_node,min_leaf,max_depth)

class QRStepTree(StepTree):
    def __init__(self,mat,rp_mat,dc_mat):
        self.root = None
        self.mat = mat
        self.rp_mat = rp_mat
        self.dc_mat = dc_mat
        self.convert_to_basis()
        self.add_root()
    
    def convert_to_basis(self):
        self.mat = _rpqr.get_qr_basis(self.mat,self.rp_mat.shape[1])
        logger.info('Using QR factorization...data matrix is '+str(self.mat.shape))

    def add_node(self,cnode,min_leaf,max_depth):

        if cnode.level < max_depth: 
            
            current_rp_mat = self.rp_mat[cnode.level,:][:,np.newaxis]

            current_mat = self.mat[cnode.indxs]
            rpvec = np.asarray(np.dot(current_mat,current_rp_mat))

            rpdcvec = rpvec
            
            pos = []
            neg = []
            for indx,val in enumerate(rpdcvec.flatten()):
                if val>= 0 : pos.append(cnode.indxs[indx])
                else : neg.append(cnode.indxs[indx])
            
            if len(pos)>0:
                new_pos_node = Node()
                new_pos_node.indxs = pos
                new_pos_node.level = cnode.level + 1
                cnode.pos_child = new_pos_node
                if len(pos)>min_leaf:
                    self.add_node(new_pos_node,min_leaf,max_depth)

            if len(neg)>0:
                new_neg_node = Node()
                new_neg_node.indxs = neg
                new_neg_node.level = cnode.level + 1
                cnode.neg_child = new_neg_node 
                if len(neg)>min_leaf:
                    self.add_node(new_neg_node,min_leaf,max_depth)
                    



