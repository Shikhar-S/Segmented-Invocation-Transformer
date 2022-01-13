class ParseTreeNode:
    def __init__(self,val):
        self.value = val
        self.children = []
        self.height = 0
        self.parent = None
        #attributes for generating S and hierarchical positional embeddings
        self.leaf_order_idx = -1 #0 indexed
        self.leaf_list = []
        
    def add_child(self,child):
        self.children.append(child)
        child.parent = self