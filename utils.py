import numpy as np
import scipy.sparse as sp
import torch
import os
import json
import pandas as pd
from scipy.sparse.linalg import spsolve
from numba import njit

def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)

def encode_onehot(labels):
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    return np.array(list(map(classes_dict.get, labels)), dtype=np.int32)

@njit
def fast_bfs_depth_limited(num_nodes, indptr, indices, is_hub_array, max_hops=3):
    rows_out = []
    cols_out = []
    queue_buffer = np.empty(num_nodes, dtype=np.int32)
    visited = np.zeros(num_nodes, dtype=np.bool_)
    
    for target in range(num_nodes):
        if is_hub_array[target]:
            rows_out.append(target)
            cols_out.append(target)
            continue
            
        visited[:] = False 
        queue_buffer[0] = target
        visited[target] = True
        
        current_level_count = 1
        next_level_count = 0
        q_start = 0
        hop = 0
        
        while current_level_count > 0 and hop < max_hops:
            for i in range(current_level_count):
                curr = queue_buffer[q_start + i]
                start_idx = indptr[curr]
                end_idx = indptr[curr+1]
                
                for k in range(start_idx, end_idx):
                    p = indices[k]
                    if not visited[p]:
                        visited[p] = True
                        if is_hub_array[p]:
                            rows_out.append(target) 
                            cols_out.append(p)
                        else:
                            if hop + 1 < max_hops:
                                queue_buffer[q_start + current_level_count + next_level_count] = p
                                next_level_count += 1
            q_start += current_level_count
            current_level_count = next_level_count
            next_level_count = 0
            hop += 1
    return rows_out, cols_out

class HierarchyProcessor:
    def __init__(self, adj, num_nodes):
        # adj: Row=Receiver (Citing), Col=Sender (Cited)
        self.adj = adj 
        self.num_nodes = num_nodes
        
    def calculate_spring_rank(self):
        adj_influence = self.adj.transpose() 

        d_out = np.array(adj_influence.sum(1)).flatten()
        d_in = np.array(adj_influence.sum(0)).flatten()
        
        M = sp.diags(d_out) + sp.diags(d_in) - (adj_influence + adj_influence.transpose()) + sp.eye(self.num_nodes) * 1e-4
        b = d_out - d_in 
        
        return spsolve(M, b)

    def identify_source_hubs(self, ranks):
        # rows接收
        # cols发出
        rows, cols = self.adj.nonzero()

        # Forward Edge: Sender Rank > Receiver Rank 
        # Backward Edge: Sender Rank < Receiver Rank、
        sender_ranks = ranks[cols]
        receiver_ranks = ranks[rows]
        
        epsilon = 1e-5 
        
        is_forward_edge = sender_ranks > (receiver_ranks + epsilon)
        is_backward_edge = sender_ranks < (receiver_ranks - epsilon)
  
        #被Forward Edge指向的节点
        nodes_directed_by_forward = np.unique(rows[is_forward_edge])
        
        # 找到被Backward Edge指向的节点
        nodes_directed_by_backward = np.unique(rows[is_backward_edge])
        
        has_incoming_forward = np.zeros(self.num_nodes, dtype=bool)
        has_incoming_backward = np.zeros(self.num_nodes, dtype=bool)
        
        has_incoming_forward[nodes_directed_by_forward] = True
        has_incoming_backward[nodes_directed_by_backward] = True
        

        #1. if v is not directed by any forward edges
        #2. else if v is directed by one or more backward edges
        #Source Hub = (No Incoming Forward) OR (Has Incoming Backward)
        
        is_hub = (~has_incoming_forward) | (has_incoming_backward)
        
        source_hubs = np.where(is_hub)[0].tolist()
        print(f"      -> Found {len(source_hubs)} Source Hubs")
        return source_hubs

    def adaptive_context_extension(self, source_hubs):
        MAX_HOPS = 4
 
        adj_search = self.adj.tocsr()
        indptr = adj_search.indptr
        indices = adj_search.indices
        is_hub_array = np.zeros(self.num_nodes, dtype=bool)
        is_hub_array[source_hubs] = True
        
        new_rows, new_cols = fast_bfs_depth_limited(self.num_nodes, indptr, indices, is_hub_array, max_hops=MAX_HOPS)
        
        orig_rows, orig_cols = self.adj.nonzero()
        
        if len(new_rows) > 0:
            all_rows = np.concatenate([orig_rows, np.array(new_rows)])
            all_cols = np.concatenate([orig_cols, np.array(new_cols)])
            data_mask = np.concatenate([np.zeros(len(orig_rows)), np.ones(len(new_rows))])
        else:
            all_rows, all_cols = orig_rows, orig_cols
            data_mask = np.zeros(len(orig_rows))

        augmented_adj = sp.coo_matrix((np.ones(len(all_rows)), (all_rows, all_cols)),
                                      shape=(self.num_nodes, self.num_nodes), dtype=np.float32)
        hub_mask = sp.coo_matrix((data_mask, (all_rows, all_cols)),
                                 shape=(self.num_nodes, self.num_nodes), dtype=np.float32)
        
        augmented_adj = augmented_adj + augmented_adj.T.multiply(augmented_adj.T > augmented_adj) - augmented_adj.multiply(augmented_adj.T > augmented_adj)
        hub_mask = hub_mask + hub_mask.T.multiply(hub_mask.T > hub_mask) - hub_mask.multiply(hub_mask.T > hub_mask)
        augmented_adj = normalize_adj(augmented_adj + sp.eye(self.num_nodes))
        
        return (torch.FloatTensor(np.array(augmented_adj.todense())), 
                torch.FloatTensor(np.array(hub_mask.todense())))

def load_data(dataset_name, root_path, split_idx=0):
    # Wiki-CS
    if dataset_name == "wiki-cs":
        import itertools
        
        if os.path.exists(os.path.join(root_path, "wiki-cs-dataset-master", "dataset", "data.json")):
            data_path = os.path.join(root_path, "wiki-cs-dataset-master", "dataset", "data.json")
        else:
            raise FileNotFoundError(
                f"Could not find data.json in {root_path}\n"
                f"Expected: {os.path.join(root_path, 'dataset', 'data.json')}"
            )
        
        with open(data_path, 'r') as f:
            data = json.load(f)

        features_raw = np.array(data['features'], dtype=np.float32)
        features = sp.csr_matrix(features_raw)
        
        labels = np.array(data['labels'], dtype=np.int64)
        
        # 转成coo形式
        edge_list = list(itertools.chain(*[
            [(i, nb) for nb in nbs] 
            for i, nbs in enumerate(data['links'])
        ]))
        edge_array = np.array(edge_list, dtype=np.int32)
        
        adj = sp.coo_matrix(
            (np.ones(len(edge_array), dtype=np.float32), 
             (edge_array[:, 0], edge_array[:, 1])),
            shape=(len(labels), len(labels)),
            dtype=np.float32
        )
  
        # 划分
        split_masks = {
            'train_mask': data['train_masks'][split_idx],
            'val_mask': data['val_masks'][split_idx],
            'stopping_mask': data['stopping_masks'][split_idx],
            'test_mask': data['test_mask']
        }
    
    
    elif dataset_name == "pubmed":
        path = os.path.join(root_path, dataset_name)
        n_file = os.path.join(path, "Pubmed-Diabetes.NODE.paper.tab")
        e_file = os.path.join(path, "Pubmed-Diabetes.DIRECTED.cites.tab")
        
        with open(n_file, 'r') as f:
            lines = f.readlines()
            
        attr_line = lines[1].strip().split('\t')
        
        vocab_map = {}
        
        for idx, attr_name in enumerate(attr_line[2:]):
            # 去掉冒号后面的类型说明
            word_key = attr_name.split(':')[0]
            vocab_map[word_key] = idx
            
        num_features = len(vocab_map)

        # 跳过header
        node_data = lines[2:] 
        num_nodes = len(node_data)
        features = sp.lil_matrix((num_nodes, num_features))
        labels = np.zeros(num_nodes, dtype=np.int64)
        idx_map = {}

        for i, line in enumerate(node_data):
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            
            paper_id = parts[0]
            idx_map[paper_id] = i
            
            # 解析标签
            labels[i] = int(parts[1].split('=')[1]) - 1
            
            # 解析特征：利用刚才建好的 vocab_map
            for feat_str in parts[2:]:
                kv = feat_str.split('=')
                if len(kv) == 2:
                    word_key = kv[0]
                    if word_key in vocab_map:
                        f_idx = vocab_map[word_key]
                        f_val = float(kv[1])
                        features[i, f_idx] = f_val
        
        features = features.tocsr()

        with open(e_file, 'r') as f:
            # 跳过前两行header
            edge_lines = f.readlines()[2:]
            
        edges = []
        for line in edge_lines:
            parts = line.strip().split('\t')
            # 格式: ID \t paper:123 \t | \t paper:456
            try:
                u_id = parts[1].split(':')[1]
                v_id = parts[3].split(':')[1]
                if u_id in idx_map and v_id in idx_map:
                    edges.append([idx_map[u_id], idx_map[v_id]])
            except:
                continue

        edges = np.array(edges)
        adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), 
                            shape=(num_nodes, num_nodes))
        
        # labels转为one-hot 
        classes = 3
        labels_onehot = np.eye(classes)[labels]
        labels = labels_onehot
        split_masks = None

    # Cora/Citeseer
    else:
        path = os.path.join(root_path, dataset_name)
        content_file = os.path.join(path, f"{dataset_name}.content")
        cites_file = os.path.join(path, f"{dataset_name}.cites")
        df_content = pd.read_csv(content_file, sep='\t', header=None, dtype=str)
        idx = df_content.iloc[:, 0].values
        labels_raw = df_content.iloc[:, -1].values
        features_raw = df_content.iloc[:, 1:-1].values.astype(np.float32)
        features = sp.csr_matrix(features_raw)
        labels = encode_onehot(labels_raw)
        idx_map = {j: i for i, j in enumerate(idx)}
        df_cites = pd.read_csv(cites_file, sep='\t', header=None, dtype=str)
        s, d = df_cites[0].map(idx_map), df_cites[1].map(idx_map)
        mask = s.notna() & d.notna()
        adj = sp.coo_matrix((np.ones(mask.sum()), (d[mask].astype(int), s[mask].astype(int))), shape=(len(labels), len(labels)))
        split_masks = None

    
    features = torch.FloatTensor(np.array(normalize_features(features).todense()))
    
    if len(labels.shape) > 1:
        labels = torch.LongTensor(np.where(labels)[1])
    else:
        labels = torch.LongTensor(labels)
    
    if split_masks is not None:
        return adj, features, labels, split_masks
    else:
        return adj, features, labels
