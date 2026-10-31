import logging

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simplified")
logger.setLevel(logging.INFO)
logger.info("logger ready")

inputs = torch.tensor([
    [0.41, 0.2, 0.89],
    [0.55, 0.83, 0.66], 
    [0.57, 0.85, 0.64],
    [0.2, 0.58, 0.3],
    [0.77, 0.24, 0.1],
    [0.05, 0.80, 0.55]
    ])

# main idea is to compute attention scores that is dot product of query with other inputs

def naive_softmax(x):
    expx = torch.exp(x)
    return expx / expx.sum()


def get_attn_scores(query, inputs, *args):

    """ return unnormalized attention scores as default """

    attn_scores = torch.empty(inputs.shape[0]) # one score per input (input[i] . input[2] 
                                           # for i in range(inputs.shape[0]))
    
    for i, x in enumerate(inputs):
        attn_scores[i] = torch.dot(query, x)

    if 'norm' in args:
        attn_scores = attn_scores / attn_scores.sum()
    elif 'softmax' in args:
        attn_scores = torch.softmax(attn_scores, dim=0)
    elif 'naive_softmax' in args:
        attn_scores = naive_softmax(attn_scores)
    else:
        attn_scores = attn_scores
    return attn_scores


def get_context_vector(query, inputs, attn_weights):
    """ context vector is weighted sum of inputs by attention scores """

    if attn_weights.sum() != 1:
        attn_weights = torch.softmax(attn_weights, dim=0)

    context_vector = torch.zeros_like(query)
    for i, x in enumerate(inputs):
        context_vector += attn_weights[i] * x
    return context_vector

def compute_all_attn_scores(inputs):
    n_inputs = inputs.shape[0]
    attn_scores = torch.empty(n_inputs, n_inputs)
    for i, x in enumerate(inputs):
        for j, y in enumerate(inputs):
            attn_scores[i, j] = torch.dot(x, y)
    return attn_scores

def compute_all_attn_scores_matrix(inputs):
    """ compute attention scores using matrix multiplication """
    return torch.matmul(inputs, inputs.T)

def compute_context_matrix(inputs, attn_weights):
    if not torch.allclose(attn_weights.sum(dim=-1), torch.ones(attn_weights.shape[0])):
        attn_weights = torch.softmax(attn_weights, dim=1)
    logger.info(f"attn_weights: {attn_weights}")
    return torch.matmul(attn_weights, inputs)



query = inputs[1]
attn_scores = compute_all_attn_scores(inputs)
attn_scores_m = compute_all_attn_scores_matrix(inputs)
#logger.info(attn_scores_m)
logger.info(attn_scores)

assert torch.allclose(attn_scores, attn_scores_m)
#--------------------------------------------------#

attn_score_normalized = get_attn_scores(query, inputs, 'softmax')
context_vector_2 = get_context_vector(query, inputs, attn_score_normalized)

context_vector = compute_context_matrix(inputs, attn_scores_m)
logger.info(context_vector)
logger.info(attn_score_normalized)
logger.info(context_vector_2)
assert torch.allclose(context_vector[1], context_vector_2)