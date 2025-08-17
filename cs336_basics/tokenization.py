import os
import time
from tqdm import tqdm
from collections import defaultdict
from typing import Iterable, Iterator
# from jaxtyping import Float, Int
import regex as re
import pickle
from typing import Generator

mini_chunk_size = 8192
pool_size = 32

def _yield_pretokens(corpus, special_tokens, include_special_tokens: bool = False) -> Generator[tuple[re.Match, str], None, None]:
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    complied_regex = re.compile(PAT)
    special_tokens_re = "|".join(re.escape(s) for s in sorted(special_tokens, key=len, reverse=True))
    combined_pattern = f"({special_tokens_re})" if special_tokens_re else None
    if combined_pattern:
        iterate_over = re.split(combined_pattern, corpus)
    else:
        iterate_over = [corpus]
    for part in iterate_over:
        if not part:  # Skip empty strings
            continue
        if include_special_tokens and part in special_tokens:
            # This is a special token, yield it as-is
            yield None, part
        else:
            for pretoken in re.finditer(complied_regex, part):
                yield pretoken, part[pretoken.start():pretoken.end()]

def _build_pretokens(corpus, special_tokens) -> dict[tuple[bytes]]:
    # print(corpus[:500] + corpus[-500:], end="\n"+"="*80+"\n")
    pretokens = defaultdict(int)
    num_special_tokens = len(special_tokens)
    for _, txt in _yield_pretokens(corpus, special_tokens):
        pretokens[tuple(d+num_special_tokens for d in txt.encode("utf-8"))] += 1
    return pretokens

def init_vocab(special_tokens):
    vocab = {i: s.encode("utf-8") for i, s in enumerate(special_tokens)}
    nxt = len(vocab)
    vocab.update({nxt + b: bytes([b]) for b in range(256)})
    return vocab

def task(args):
    input_path, special_tokens, st, end = args
    fp = open(input_path, 'rb')
    fp.seek(st)
    corp = fp.read(end-st).decode("utf-8")
    pretoken = _build_pretokens(corp, special_tokens)
    return pretoken

def build_pretokens_chunked(input_path, special_tokens):
    # DO NOT load full file in memory at once
    fp = open(input_path, 'rb')
    fp.seek(0, os.SEEK_END)
    file_size = fp.tell()
    fp.seek(0)
    num_processes = pool_size
    chunk_size = file_size // num_processes
    print(f"{chunk_size=}")
    num_chunks = file_size // chunk_size + 1
    chunk_boundaries = [i*chunk_size for i in range(num_chunks+1)]
    chunk_boundaries[-1] = file_size
    for bi in range(1, len(chunk_boundaries)-1):
        curr_pos = chunk_boundaries[bi]
        fp.seek(curr_pos)
        while True:
            mini_chunk = fp.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                chunk_boundaries = chunk_boundaries[:bi + 1]
                break
            ret = mini_chunk.find(special_tokens[0].encode('utf-8'))
            if ret != -1:
                chunk_boundaries[bi] = curr_pos + ret
                break
            curr_pos += len(mini_chunk)
        if chunk_boundaries[bi] == file_size:
            break
    pretokens_final = defaultdict(int)
    
    import multiprocessing as mp
    with mp.Pool(processes=pool_size) as pool:
        # Use imap_unordered to process results as they become available
        results = pool.imap_unordered(task, [(input_path, special_tokens, chunk_boundaries[i], chunk_boundaries[i+1]) for i in range(len(chunk_boundaries) -1)])
        completed = 0
        # Update pretokens_final as each process completes
        for pretoken in results:
            completed += 1
            print(f"{completed}/{num_processes}")
            for p, cnt in pretoken.items():
                pretokens_final[p] += cnt
    
    return pretokens_final

def build_pretokens(input_path, special_tokens):
    corpus = open(input_path, 'r').read()
    return _build_pretokens(corpus, special_tokens)


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str] | None = None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = init_vocab(special_tokens)    
    assert len(special_tokens) == 1
    pretoken_time = time.time()
    pretokens = build_pretokens_chunked(input_path, special_tokens)
    pretoken_time = time.time() - pretoken_time
    print(f"{pretoken_time=}")
    def build_pairs():
        pairs = defaultdict(int)
        for pretoken, cnt in pretokens.items():
            for i in range(len(pretoken) - 1):
                pairs[pretoken[i], pretoken[i+1]] += cnt
        return pairs

    def update_pretokens():
        delete_keys = []
        to_add = {}
        for pretoken, cnt in pretokens.items():
            if max_key[0] not in pretoken or max_key[1] not in pretoken:
                continue
            t = 0
            token_list = pretoken
            new_token_list = []
            while t < len(token_list):
                if t + 1 < len(token_list) and (token_list[t], token_list[t+1]) == max_key:
                    new_token_list.append(new_token_idx)
                    t += 1
                else:
                    new_token_list.append(token_list[t])
                t += 1

            if len(token_list) == len(new_token_list):
                continue

            for i in range(len(new_token_list)-1):
                pairs[(new_token_list[i], new_token_list[i+1])] += cnt
            for i in range(len(token_list) - 1):
                pairs[(token_list[i], token_list[i+1])] -= cnt
                if not pairs[(token_list[i], token_list[i+1])]:
                    del pairs[(token_list[i], token_list[i+1])]
            delete_keys.append(pretoken)
            to_add[tuple(new_token_list)] = cnt

        for d in delete_keys:
            pretokens.pop(d)
        pretokens.update(to_add)
        token_list = new_token_list

    merges = []
    # tqdm progress bar for merge loop
    pbar = tqdm(total=vocab_size - len(vocab), desc="BPE merges")
    pairs = build_pairs()
    while len(vocab) < vocab_size:
        max_key = max(pairs.keys(), key=lambda x: (pairs[x], vocab[x[0]], vocab[x[1]]))
        new_token = vocab[max_key[0]] + vocab[max_key[1]]
        merges.append((vocab[max_key[0]], vocab[max_key[1]]))
        new_token_idx = len(vocab)
        vocab[new_token_idx] = new_token
        update_pretokens()
        pbar.update(1)
    pbar.close()
    return vocab, merges


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.inv_vocab: dict[bytes, int] = {v: i for i, v in self.vocab.items()}
        self.inv_merge: dict[tuple[int, int], int] = {(self.inv_vocab[v[0]], self.inv_vocab[v[1]]): self.inv_vocab[v[0] + v[1]] for i, v in enumerate(self.merges)}

    def encode(self, text: str) -> list[int]:
        ans = []
        for pretoken, txt in _yield_pretokens(text, self.special_tokens, include_special_tokens=True):
            if txt in self.special_tokens:
                ans.append(self.inv_vocab[txt.encode("utf-8")])
                continue
            txt = txt.encode("utf-8", errors="ignore")
            tokens = [self.inv_vocab[bytes([t])] for t in  txt]
            while True:
                final = []
                min_v = len(self.vocab) + 1
                for i in range(len(tokens) - 1):
                    if (curr_v:= self.inv_merge.get((tokens[i], tokens[i+1]), len(self.vocab) + 2)) < min_v:
                        min_v = curr_v
                if min_v == len(self.vocab) + 1:
                    break
                i = 0
                while i < len(tokens):
                    if i+1 < len(tokens) and (curr_v:= self.inv_merge.get((tokens[i], tokens[i+1]), len(self.vocab) + 2)) == min_v:
                        final.append(min_v)
                        i += 2
                    else:
                        final.append(tokens[i])
                        i += 1
                tokens = final
            ans.extend(tokens)
        return ans

    def decode(self, tokens: list[int]) -> str:
        ans = [self.vocab[t] for t in tokens]
        return b"".join(ans).decode("utf-8", errors="replace")

    def encode_iterable(self, f: Iterable[str], chunk_size: int = 524288) -> Iterator[int]:
        lingering_chunks = []
        # create a tqdm progress bar
        # total file size
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)
        pbar = tqdm(total=file_size, desc="Encoding")
        pbar.update(0)
        total_done = 0
        if not self.special_tokens:
            yield from self.encode(f.read())
            return
        assert len(self.special_tokens) == 1
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            lingering_chunks.append(chunk)
            lingering_text = "".join(lingering_chunks)
            ret = lingering_text.rfind(self.special_tokens[0])

            if ret != -1:
                position = ret + len(self.special_tokens[0])
                text_to_tokenize = lingering_text[:position]
                lingering_chunks = [lingering_text[position:]]
                # print(text_to_tokenize, end="\n"+"="*80+"\n")
                yield from self.encode(text_to_tokenize)
                total_done += len(text_to_tokenize)
                pbar.update(len(text_to_tokenize))
            else:
                lingering_chunks.append(chunk)

        if lingering_chunks:
            text_to_tokenize = "".join(lingering_chunks)
            yield from self.encode(text_to_tokenize)
            total_done += len(text_to_tokenize)
            pbar.update(len(text_to_tokenize))
        pbar.close()
        return

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        vocab = pickle.load(open(vocab_path, "rb"))
        merges = pickle.load(open(merges_path, "rb"))
        return cls(vocab, merges, special_tokens)

if __name__ == "__main__":
    vocab_path = "data/vocab_TinyStories_final"
    merges_path = "data/merges_TinyStories_final"
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    ret = tokenizer.encode(" the")
    print(ret)
    assert len(tokenizer.vocab) == 10000

