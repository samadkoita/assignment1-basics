# Byte-Pair Encoding (BPE) Tokenizer

## The Unicode Standard

What Unicode character does chr(0) return?
- The unicode character +U0000. Escape Sequence in Python '\x00'
- It is a NULL character - does not print anything. In C used to seperate strings in raw memory (eg: in a string array)
- In python, you can include these as part of your string. In C, including this may cause issues as it may be used to indicate end of a string

How does this character’s string representation (__repr__()) differ from its printed representation?
- str(chr(0)) = ''; repr(chr(0)) = '\x00'

What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
- It is not visible in printed representation of the text, but in reality is present

## Unicode Encodings

What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.
- Same content is represented in less tokens. This has advantages in efficiency and learning:
Efficiency:
- Shorter sequences - less FLOPS, larger context window, 
- better training efficiency: Higher effective batch size given fixed compute. Models can therefore see more diverse examples per batch.
- Worse learning:
    Gradient Flows: o = q.k v + q. k. v such that sum(q.k) = 1. If there are more q.k, each q.k is smaller. so less gradient signals flows through each part of computational graph
    Compounding errors:
    Less layers to represent a specific concept: (eg: photon v/s p-h-o-t-o-n). If p, h, o, t, o, n had different token embeddings, it would take at least 1 layer for the tokens to attend to each other, etc. Model can learn common concepts at the token embedding level itself and focus its computation to reason about the rest of the sentence, 
    Block size during training: In each block, you can pack a lot more "effective context". Specially relevant for pretraining when you have to break up the data into blocks
    Long range reasoning: Model has effectively much larger context window
    http://claude.ai/chat/fd3403bd-9d41-46e5-8c70-321fa3b20308

Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
"hello\u0394"

Why? In UTF-8, a single unicode character can be represented by 2 bytes. When you iterate on the bytes string, the second byte alone cannot be directly decoded as it does not map to a complete unicode value

Give a two byte sequence that does not decode to any Unicode character(s).
bytes = b'\xFF\xFF' [by utf-8]
This is because it cannot be interpreted as 2 1 byte or a single 2 byte char



## Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)
(b) What part of tokenizer takes most time?

## Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)
(b) What part of tokenizer takes most time?
chunk_size=69617286
pool_size = 32
pretoken = 41.01s
merged = 2 mins

## Problem (tokenizer): Implementing the tokenizer (15 points)


## Problem (tokenizer_experiments): Experiments with tokenizers (4 points)
(a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?
TinyStories: 4.36 bytes/token
OWT: 4.16 bytes / token

(b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.
TinyStories Tokenizer on OWT: 3.37 bytes/token
Much more tokens to encode same info - less efficient compression

(c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?
1.13 MB / sec - encode
0.78 MB / sec - encode iterable

(d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We’ll use this later to train our language model. We recommend serializing the token IDs as a NumPy array of datatype uint16. Why is uint16 an appropriate choice?
Done for TinyStories
uint16 is sufficient to represent 32K > 2^5*2^10 = 15 bits



## Resource Accounting


## Problem (learning_rate_tuning): Tuning the learning rate (1 point)
As we will see, one of the hyperparameters that affects training the most is the learning rate. Let’s
see that in practice in our toy example. Run the SGD example above with three other values for the
learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss for each
of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over the course of
training)?
loss = (weights**2).mean() -> `gradient = 2*w`
data -= 2*w * lr / Count(w)
if lr > count(weights) -> diverges
if lr < count(weights) -> converges

