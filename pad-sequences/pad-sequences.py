import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    padded_result = []
    if(max_len is None):
        max_len = 0
        for seq in seqs:
            max_len = max(max_len, len(seq))

    for seq in seqs:
        if(len(seq) < max_len):
            padded_seq = seq.copy()
            current_length = len(seq)
            padded_seq.extend([pad_value for i in range(max_len - current_length)])

        else:
            padded_seq = [seq[i] for i in range(max_len)]
        padded_result.append(np.array(padded_seq))
    return np.array(padded_result)
        
            