# main difference between simpliefied attention and self attention is that self attention uses 
# queries, keys and values matrices
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch 



class SelfAttention_v1(torch.nn.Module):
    def __init__(self, d_in, d_out): 
        super().__init__()

        #self.W_q = torch.nn.Parameter(torch.randn(d_in, d_out))
        #self.W_k = torch.nn.Parameter(torch.randn(d_in, d_out))
        #self.W_v = torch.nn.Parameter(torch.randn(d_in, d_out))
        # a better way to init params is to use torch.nn.Linear instead of raw parameters
        self.W_q = torch.nn.Linear(d_in, d_out)
        self.W_k = torch.nn.Linear(d_in, d_out)
        self.W_v = torch.nn.Linear(d_in, d_out)
        self.d_out = d_out
        self.assign_params()



    def assign_params(self):
        # to assign weights from torch.nn.Linear we can extract them as 
        q_weight = self.W_q.weight.detach().clone().T
        k_weight = self.W_k.weight.detach().clone().T
        v_weight = self.W_v.weight.detach().clone().T
        
        self.q_bias = torch.nn.Parameter(self.W_q.bias.detach().clone(), requires_grad=True)
        self.k_bias = torch.nn.Parameter(self.W_k.bias.detach().clone(), requires_grad=True)
        self.v_bias = torch.nn.Parameter(self.W_v.bias.detach().clone(), requires_grad=True)

        # now lets initialize torch.nn.parameters with these weights
        self.W_q_params = torch.nn.Parameter(q_weight, requires_grad=True)
        self.W_k_params = torch.nn.Parameter(k_weight, requires_grad=True)
        self.W_v_params = torch.nn.Parameter(v_weight, requires_grad=True)

        

    def forward_with_parameters_old(self,x):
        q = torch.matmul(x, self.W_q_params) + self.q_bias
        k = torch.matmul(x, self.W_k_params) + self.k_bias
        v = torch.matmul(x, self.W_v_params) + self.v_bias

        attn_scores = torch.matmul(q, k.T)
        attn_weights = torch.softmax(attn_scores / torch.sqrt(torch.tensor(self.d_out)), dim=1)
        context_vec = torch.matmul(attn_weights, v)
        return context_vec

    def forward(self, x):

        #q = torch.matmul(x, self.W_q)
        #k = torch.matmul(x, self.W_k)
        #v = torch.matmul(x, self.W_v)
        
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        attn_scores = torch.matmul(q, k.T)
        attn_weights = torch.softmax(attn_scores / torch.sqrt(torch.tensor(self.d_out)), dim=1)
        context_vec = torch.matmul(attn_weights, v)
        return context_vec 


torch.manual_seed(123)
inputs = torch.tensor([
    [0.41, 0.2, 0.89],
    [0.55, 0.83, 0.66], 
    [0.57, 0.85, 0.64],
    [0.2, 0.58, 0.3],
    [0.77, 0.24, 0.1],
    [0.05, 0.80, 0.55]
    ])
# input size is n x d_in 
d_in = inputs.shape[1]
d_out = 2 
self_attn = SelfAttention_v1(d_in, d_out)
context = self_attn(inputs)
context_2  = self_attn.forward_with_parameters_old(inputs)
logger.info(f"context: {context}")
logger.info(f"context_2: {context_2}")

def rough():

    inputs = torch.tensor([
        [0.41, 0.2, 0.89],
        [0.55, 0.83, 0.66], 
        [0.57, 0.85, 0.64],
        [0.2, 0.58, 0.3],
        [0.77, 0.24, 0.1],
        [0.05, 0.80, 0.55]
        ])
    # input size is n x d_in 

    query = inputs[1]
    d_in = inputs.shape[1]
    d_out = 2 

    W_q = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False) # Q: what does torch.nn.Parameter do?
    W_k = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
    W_v = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)

    # W_* size is d_in x d_out

    # multiple query with each matrix
    q = torch.matmul(query, W_q)  # q size here is 1 x d_out
    k = torch.matmul(query, W_k)
    v = torch.matmul(query, W_v)
    logger.info(f"q: {q}, k: {k}, v: {v}")

    # to get these values for each input
    queries = torch.matmul(inputs, W_q) # size n x d_out
    keys = torch.matmul(inputs, W_k) # size n x d_out
    values = torch.matmul(inputs, W_v) # size n x d_out
    logger.info(f"queries: {queries}, keys: {keys}, values: {values}")

    # main point now is to compute attention scores we will multiple queries with keys
    attn_scores = torch.matmul(queries, keys.T)  # size is n x n 
    logger.info(f"attn_scores: {attn_scores}")

    # and to normalize we take softmax and divide by square root of d_out 
    # wrong way -> 
    # attn_scores = torch.softmax(attn_scores, dim = 1) / (d_out ** 0.5)  # attn score size is n x n
    # logger.info(f"attn_scores v1: {attn_scores}")

    attn_scores = torch.softmax(attn_scores / (d_out ** 0.5), dim = 1)    # attn score size is n x n

    # next context vector for each input is weighted sum of values weighted by attn scores
    context = torch.matmul(attn_scores, values) # size n x d_out

    logger.info(f"attn_scores: {attn_scores}")
    logger.info(f"context: {context}")

# revise 
# x gets multiplied with W_q, W_k, W_v giving us n x d_out matrices
# attention scores are normalized product of queries and keys that is n x n matrix
# context is then attn_scores times values giving n x d_out matrix

# now we implement a compact self attention class
