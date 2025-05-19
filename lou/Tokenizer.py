# Sub library used for tokenized a sentence. We will start simple with character to ascii tokenization.
# We will then feed the tokenized caharacters into a Neural Network with a loop in function of how much entry the said network can support

def CharTokenizer(sentence):
    return [ord(c) for c in list(sentence)]