import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import List, Dict, Optional
import math
from collections import Counter
import logging
import random

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class ModelConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get('vocab_size', 1024)
        self.d_model = kwargs.get('d_model', 256)
        self.n_heads = kwargs.get('n_heads', 8)
        self.n_groups = kwargs.get('n_groups', 4)
        self.n_layers = kwargs.get('n_layers', 4)
        self.d_ff = kwargs.get('d_ff', 1024)
        self.max_seq_len = kwargs.get('max_seq_len', 128)
        self.dropout = kwargs.get('dropout', 0.1)
        self.n_experts = kwargs.get('n_experts', 1)
        self.top_k_experts = kwargs.get('top_k_experts', 1)
        self.beam_width = kwargs.get('beam_width', 5)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

class BPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.word_to_idx = {'<UNK>': 0, '<END>': 1, '<PAD>': 2, '<SEP>': 3}
        self.idx_to_word = {0: '<UNK>', 1: '<END>', 2: '<PAD>', 3: '<SEP>'}
        self.merges = []

    def _get_pairs(self, words: Dict[str, int]) -> Counter:
        pairs = Counter()
        for word, freq in words.items():
            chars = list(word)
            for i in range(len(chars) - 1):
                pairs[(chars[i], chars[i + 1])] += freq
        return pairs

    def _merge_pair(self, pair: tuple, words: Dict[str, int]) -> Dict[str, int]:
        new_words = {}
        for word, freq in words.items():
            new_word = ''
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word += pair[0] + pair[1]
                    i += 2
                else:
                    new_word += word[i]
                    i += 1
            new_words[new_word] = new_words.get(new_word, 0) + freq
        return new_words

    def train(self, corpus: List[tuple], iterations: int = 200) -> None:
        if not corpus:
            raise ValueError("Corpus cannot be empty.")
        
        words = [word.lower() for qa in corpus for text in qa for word in text.split()]
        word_freq = Counter(words)
        for word in word_freq.keys():
            if word not in self.word_to_idx and len(self.word_to_idx) < self.vocab_size:
                self.word_to_idx[word] = len(self.word_to_idx)
                self.idx_to_word[len(self.idx_to_word)] = word
        
        current_words = word_freq.copy()
        for _ in range(iterations):
            pairs = self._get_pairs(current_words)
            if not pairs or len(self.word_to_idx) >= self.vocab_size:
                break
            best_pair = pairs.most_common(1)[0][0]
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.word_to_idx:
                self.merges.append(best_pair)
                self.word_to_idx[new_token] = len(self.word_to_idx)
                self.idx_to_word[len(self.idx_to_word)] = new_token
                current_words = self._merge_pair(best_pair, current_words)
        
        while len(self.word_to_idx) < self.vocab_size:
            dummy = f'<DUMMY_{len(self.word_to_idx)}>'
            self.word_to_idx[dummy] = len(self.word_to_idx)
            self.idx_to_word[len(self.idx_to_word)] = dummy
        logger.info(f"Tokenizer trained. Vocabulary size: {len(self.word_to_idx)}")

    def tokenize(self, text: str, max_len: int) -> torch.Tensor:
        words = text.lower().split()
        tokens = []
        for word in words:
            current = word
            for pair in self.merges:
                new_word = ''
                i = 0
                while i < len(current):
                    if i < len(current) - 1 and (current[i], current[i + 1]) == pair:
                        new_word += pair[0] + pair[1]
                        i += 2
                    else:
                        new_word += current[i]
                        i += 1
                current = new_word
            tokens.append(self.word_to_idx.get(current, self.word_to_idx['<UNK>']))
        
        if len(tokens) >= max_len - 1:
            tokens = tokens[:max_len - 1]
        tokens.append(self.word_to_idx['<END>'])
        tokens += [self.word_to_idx['<PAD>']] * (max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def tokenize_qa(self, question: str, answer: str, max_len: int) -> torch.Tensor:
        q_tokens = self.tokenize(question, max_len // 2)[:-1]  # Remove <END>
        a_tokens = self.tokenize(answer, max_len // 2)[:-1]  # Remove <END>
        sep_token = torch.tensor([self.word_to_idx['<SEP>']], dtype=torch.long)
        combined = torch.cat([q_tokens, sep_token, a_tokens, torch.tensor([self.word_to_idx['<END>']])])
        if combined.shape[0] > max_len:
            combined = combined[:max_len - 1]
            combined = torch.cat([combined, torch.tensor([self.word_to_idx['<END>']])])
        padded = torch.cat([combined, torch.full((max_len - combined.shape[0],), self.word_to_idx['<PAD>'], dtype=torch.long)])
        return padded[:max_len]

class RotaryEmbedding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        theta = 1.0 / (10000 ** (torch.arange(0, d_model // 2).float() / (d_model // 2))).to(device)
        self.register_buffer('theta', theta)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=device).unsqueeze(1)
        angle_rates = positions * self.theta.unsqueeze(0)
        cos, sin = torch.cos(angle_rates), torch.sin(angle_rates)
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        x_rot = torch.cat([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)
        return x_rot

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_groups: int, dropout: float, max_seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_groups == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.d_k = d_model // n_heads
        self.d_v = self.d_k
        self.heads_per_group = n_heads // n_groups

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, self.n_groups * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.n_groups * self.d_v, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(d_model, max_seq_len)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = X.shape
        X = self.rope(X)
        Q = self.W_q(X).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(X).view(batch_size, seq_len, self.n_groups, self.d_k).transpose(1, 2)
        V = self.W_v(X).view(batch_size, seq_len, self.n_groups, self.d_v).transpose(1, 2)

        Q = Q.view(batch_size, self.n_groups, self.heads_per_group, seq_len, self.d_k)
        scores = torch.einsum('bghsd,bgtd->bghst', Q, K) / math.sqrt(self.d_k)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1) * -1e9
        scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.einsum('bghst,bgtd->bghsd', attn_weights, V)
        attn_output = attn_output.reshape(batch_size, self.n_heads, seq_len, self.d_v).transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.W_o(attn_output)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_model, d_ff)
        self.W3 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.W1(x))
        x = self.W2(x) * gate
        return self.W3(x)

class MoEFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int = 1, top_k: int = 1):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([SwiGLU(d_model, d_ff) for _ in range(n_experts)])
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        router_logits = self.router(x)
        top_k_logits, top_k_indices = router_logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[..., k]
            weight = top_k_weights[..., k].unsqueeze(-1)
            for i in range(batch_size):
                for j in range(seq_len):
                    expert = self.experts[expert_idx[i, j]]
                    output[i, j] = output[i, j] + weight[i, j] * expert(x[i, j].unsqueeze(0)).squeeze(0)
        return output

class TransformerLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(config.d_model, config.n_heads, config.n_groups, config.dropout, config.max_seq_len)
        self.norm2 = RMSNorm(config.d_model)
        self.ff = MoEFeedForward(config.d_model, config.d_ff, config.n_experts, config.top_k_experts)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

class AdvancedLLM(nn.Module):
    def __init__(self, config: ModelConfig, tokenizer: Optional[BPETokenizer] = None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer if tokenizer else BPETokenizer(config.vocab_size)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(config.d_model, config.max_seq_len)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.out_dropout = nn.Dropout(config.dropout)
        self.out = nn.Linear(config.d_model, config.vocab_size)
        self.to(device)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        x = self.rope(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.out_dropout(x)
        return self.out(x)

    def fit(self, qa_pairs: List[tuple], epochs: int = 50, learning_rate: float = 3e-4, batch_size: int = 2):
        self.tokenizer.train(qa_pairs)
        optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        dataset = torch.stack([self.tokenizer.tokenize_qa(q, a, self.config.max_seq_len) for q, a in qa_pairs])
        dataloader = DataLoader(TensorDataset(dataset), batch_size=batch_size, shuffle=True)
        logger.info(f"Dataset prepared. Number of samples: {len(dataset)}, Batches: {len(dataloader)}")

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                batch = batch[0].to(device)
                logger.debug(f"Batch {batch_idx} shape: {batch.shape}")

                sep_idx = (batch == self.tokenizer.word_to_idx['<SEP>']).nonzero(as_tuple=True)[1]
                logger.debug(f"Sep indices: {sep_idx.tolist()}")

                input_tokens = batch[:, :-1]
                target = batch[:, 1:]

                optimizer.zero_grad()
                logits = self.forward(input_tokens)
                logger.debug(f"Logits shape: {logits.shape}, Target shape: {target.shape}")

                mask = torch.zeros_like(target, dtype=torch.bool)
                for i in range(batch.shape[0]):
                    mask[i, sep_idx[i]:] = True

                if mask.sum() == 0:
                    logger.warning(f"Mask is empty for batch {batch_idx}, skipping")
                    continue

                ce_loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.word_to_idx['<PAD>'])
                logits_flat = logits.contiguous().view(-1, self.config.vocab_size)
                target_flat = target.contiguous().view(-1)
                mask_flat = mask.view(-1)
                loss = ce_loss(logits_flat[mask_flat], target_flat[mask_flat])
                logger.debug(f"Loss: {loss.item()}")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(dataloader) if total_loss > 0 else float('nan')
            logger.info(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    def generate(self, question: str, max_length: int = 50, beam_width: int = 5) -> str:
        self.eval()
        # Tokenize question without <END> or padding initially
        q_tokens = self.tokenize_question(question)
        sep_token = torch.tensor([self.tokenizer.word_to_idx['<SEP>']], dtype=torch.long)
        initial_tokens = torch.cat([q_tokens, sep_token]).unsqueeze(0).to(device)
        beam_scores = torch.zeros((1, beam_width), device=device)
        beam_tokens = initial_tokens.repeat(beam_width, 1)
        
        logger.debug(f"Initial tokens for generation: {initial_tokens.tolist()}")
        with torch.no_grad():
            for step in range(max_length - initial_tokens.shape[1]):
                logits = self.forward(beam_tokens)
                logits = logits[:, -1, :]
                logits[:, self.tokenizer.word_to_idx['<UNK>']] = -float('inf')
                logits[:, self.tokenizer.word_to_idx['<PAD>']] = -float('inf')
                if step < 5:  # Reduced penalty duration
                    logits[:, self.tokenizer.word_to_idx['<END>']] -= 10.0  # Reduced penalty
                next_probs = F.log_softmax(logits, dim=-1)  # Removed temperature scaling for simplicity
                
                logger.debug(f"Step {step}, Top 5 probs: {torch.topk(next_probs[0], 5).indices.tolist()}")

                vocab_size = self.config.vocab_size
                next_scores = beam_scores.unsqueeze(-1) + next_probs
                next_scores = next_scores.view(-1)
                top_k_scores, top_k_tokens = next_scores.topk(beam_width, dim=0)
                
                length_penalty = ((step + 1 + initial_tokens.shape[1]) / (initial_tokens.shape[1])) ** 0.8
                top_k_scores = top_k_scores / length_penalty
                
                beam_indices = top_k_tokens // vocab_size
                token_indices = top_k_tokens % vocab_size
                beam_tokens = torch.cat([beam_tokens[beam_indices], token_indices.unsqueeze(-1)], dim=1)
                beam_scores = top_k_scores.unsqueeze(0)
                
                if (beam_tokens[:, -1] == self.tokenizer.word_to_idx['<END>']).all():
                    logger.debug(f"Generation terminated at step {step}")
                    break
        
        best_beam = beam_tokens[0]
        logger.info(f"Raw output tokens: {best_beam.tolist()}")
        
        sep_idx = (best_beam == self.tokenizer.word_to_idx['<SEP>']).nonzero(as_tuple=True)[0].item()
        answer_tokens = best_beam[sep_idx + 1:]
        output_words = [self.tokenizer.idx_to_word.get(t.item(), '<UNK>') for t in answer_tokens]
        return ' '.join(word for word in output_words if word not in ['<PAD>', '<END>'])

    def tokenize_question(self, question: str) -> torch.Tensor:
        """Tokenize question without adding <END> or padding."""
        words = question.lower().split()
        tokens = []
        for word in words:
            current = word
            for pair in self.tokenizer.merges:
                new_word = ''
                i = 0
                while i < len(current):
                    if i < len(current) - 1 and (current[i], current[i + 1]) == pair:
                        new_word += pair[0] + pair[1]
                        i += 2
                    else:
                        new_word += current[i]
                        i += 1
                current = new_word
            tokens.append(self.tokenizer.word_to_idx.get(current, self.tokenizer.word_to_idx['<UNK>']))
        return torch.tensor(tokens, dtype=torch.long)

def get_default_qa_corpus() -> List[tuple]:
    return [
        ("What is the tallest mountain in the world, and where is it located geographically speaking?", 
         "The tallest mountain in the world is Mount Everest, standing at 8,848 meters above sea level. It is located in the Himalayas on the border between Nepal and the Tibet Autonomous Region of China."),
        ("How do bees contribute to the ecosystem, and why are they considered vital?", 
         "Bees are vital pollinators, transferring pollen between flowers to enable plant reproduction. They support the growth of fruits, vegetables, and nuts, sustaining biodiversity and food chains."),
    ] * 20

def main():
    config = ModelConfig()
    config.update(d_model=256, n_layers=4, vocab_size=1024, max_seq_len=128)

    qa_corpus = get_default_qa_corpus()
    logger.info(f"Corpus size: {len(qa_corpus)} QA pairs")

    logger.info("Initializing model...")
    model = AdvancedLLM(config)

    logger.info("Starting training...")
    model.fit(qa_corpus, epochs=5, learning_rate=3e-4, batch_size=2)  # Increased epochs

    prompts = [
        "What is the tallest mountain in the world, and where is it located geographically speaking?",
        "How do bees contribute to the ecosystem, and why are they considered vital?",
    ]

    logger.info("\nTesting the model:")
    for prompt in prompts:
        response = model.generate(prompt, max_length=50, beam_width=config.beam_width)
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Response: {response}\n")

if __name__ == "__main__":
    main()
