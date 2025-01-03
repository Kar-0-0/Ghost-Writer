import torch
import torch.nn as nn 
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

# Hyperparameters -----------
device = 'mps'
context_length = 8
n_emb = 64
n_heads = 8
dropout = 0.2
head_size = n_emb // n_heads
# ---------------------------

print(f'Using {device}')

ds = load_dataset("brunokreiner/genius-lyrics")

dataset = ''.join(lyric for lyric in tqdm(ds['train']['lyrics'], desc="Processing lyrics"))
vocab = sorted(list(set(dataset)))
vocab_len = len(vocab)

train = dataset[0:(len(dataset)*9)//10]
val = dataset[(len(dataset)*9)//10:]

stoi = {s:i for i, s in enumerate(vocab)}
itos = {i:s for i, s in stoi.items()}

def encode(sx): return torch.tensor([stoi[x] for x in sx], dtype=torch.int32, device=device)
def decode(ix): return [itos[x] for x in ix]

train = encode(train)
val = encode(val)


def get_batch(type, batch_size, context_length):
    assert type == 'train' or type == 'val'
    if type == 'train':
        batch = torch.randint(0, len(train) - context_length, (batch_size,))
    else:
        batch = torch.randint(0, len(val) - context_length, (batch_size,))

    x = train if type == 'train' else val

    x = torch.stack([train[b:b+context_length] for b in batch])
    y = torch.stack([train[b+1:b+context_length+1] for b in batch])
    
    return x, y


class SelfAttentionHead(nn.Module):
    def __init__(self, emb_dim, head_size):
        super().__init__()
        self.query = nn.Linear(emb_dim, head_size, bias=False, device=device)
        self.key = nn.Linear(emb_dim, head_size, bias=False, device=device)
        self.value = nn.Linear(emb_dim, head_size, bias=False, device=device)
        self.register_buffer('tril', torch.ones(context_length, context_length, device=device))

    def forward(self, x):
        _, T, _ = x.shape

        q = self.query(x) # (B, T, C)
        k = self.key(x)   # (B, T, C)
        v = self.value(x) # (B, T, C)
        
        scores = q @ k.transpose(-2, -1) # (B, T, C) @ (B, C, T) ---> (B, T, T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)
        
        output = scores @ v
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.head_size = n_emb // n_heads
        self.heads = nn.ModuleList([SelfAttentionHead(n_emb, self.head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_emb, n_emb)
    
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads])
        x = self.proj(x)
        
        return x


class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4*n_emb, device=device),
            nn.ReLU(), 
            nn.Linear(4*n_emb, n_emb, device=device),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_heads, n_emb):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_emb)
        self.sa = MultiHeadAttention(n_heads)
        self.ln2 = nn.LayerNorm(n_emb)
        self.ffwd = FeedForward(n_emb)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x
    

class LyricModel(nn.Module):
    def __init__(self, n_layers, n_heads, n_emb):
        super().__init__()
        self.tok_emb_table = nn.Embedding(vocab_len, n_emb)
        self.pos_emb_table = nn.Embedding(context_length, n_emb)
        self.blocks = nn.Sequential(*[Block(n_heads, n_emb) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_emb)
        self.fcl = nn.Linear(n_emb, vocab_len)
    
    def forward(self, x):
        _, T = x.shape
        tok_emb = self.tok_emb_table(x) # (B, T) ---> (B, T, C)
        positions = torch.arange(T, device=device)
        pos_emb = self.pos_emb_table(positions)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        x = self.fcl(x)
        x = F.softmax(x)

        return x

    def generate(self, idx, characters):
        pass