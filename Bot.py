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
        self.d_model = kwargs.get('d_model', 512)  # Increased for richer embeddings
        self.n_heads = kwargs.get('n_heads', 16)  # More heads for better attention
        self.n_groups = kwargs.get('n_groups', 4)
        self.n_layers = kwargs.get('n_layers', 12)  # More layers for capacity
        self.d_ff = kwargs.get('d_ff', 2048)  # Larger feedforward network
        self.max_seq_len = kwargs.get('max_seq_len', 128)
        self.dropout = kwargs.get('dropout', 0.1)
        self.n_experts = kwargs.get('n_experts', 4)  # More experts in MoE
        self.top_k_experts = kwargs.get('top_k_experts', 2)
        self.beam_width = kwargs.get('beam_width', 10)  # Increased for better generation

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
        q_tokens = self.tokenize(question, max_len // 2)[:-1]
        a_tokens = self.tokenize(answer, max_len // 2)[:-1]
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

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1) * -1e9
        scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(attn_output)

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
    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int):
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
        self.self_attn = MultiHeadSelfAttention(config.d_model, config.n_heads, config.dropout)
        self.norm2 = RMSNorm(config.d_model)
        self.gqa = GroupedQueryAttention(config.d_model, config.n_heads, config.n_groups, config.dropout, config.max_seq_len)
        self.norm3 = RMSNorm(config.d_model)
        self.ff = MoEFeedForward(config.d_model, config.d_ff, config.n_experts, config.top_k_experts)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x)))  # Added self-attention
        x = x + self.dropout(self.gqa(self.norm2(x)))
        x = x + self.dropout(self.ff(self.norm3(x)))
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

    def fit(self, qa_pairs: List[tuple], epochs: int = 30, learning_rate: float = 3e-4, batch_size: int = 16):
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

                # Masks for question and answer portions
                mask_answer = torch.zeros_like(target, dtype=torch.bool)
                mask_question = torch.zeros_like(target, dtype=torch.bool)
                for i in range(batch.shape[0]):
                    mask_answer[i, sep_idx[i]:] = True  # Answer portion
                    mask_question[i, :sep_idx[i]] = True  # Question portion

                ce_loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.word_to_idx['<PAD>'])
                logits_flat = logits.contiguous().view(-1, self.config.vocab_size)
                target_flat = target.contiguous().view(-1)
                
                # Compute loss: full weight on answer, 0.1 weight on question for context
                loss_answer = ce_loss(logits_flat[mask_answer.view(-1)], target_flat[mask_answer.view(-1)])
                loss_question = ce_loss(logits_flat[mask_question.view(-1)], target_flat[mask_question.view(-1)])
                loss = loss_answer + 0.1 * loss_question  # Weighted sum
                logger.debug(f"Loss: {loss.item()} (Answer: {loss_answer.item()}, Question: {loss_question.item()})")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    def generate(self, question: str, max_length: int = 50, beam_width: int = 10, top_k: int = 40) -> str:
        self.eval()
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
                if step < 5:
                    logits[:, self.tokenizer.word_to_idx['<END>']] -= 10.0
                
                # Combine beam search with top-k sampling
                next_probs = F.log_softmax(logits, dim=-1)
                top_k_probs, top_k_indices = next_probs.topk(top_k, dim=-1)
                next_probs = torch.zeros_like(next_probs).scatter_(-1, top_k_indices, top_k_probs)
                next_probs = next_probs / next_probs.sum(dim=-1, keepdim=True)  # Renormalize
                
                logger.debug(f"Step {step}, Top 5 probs: {torch.topk(next_probs[0], 5).indices.tolist()}")

                vocab_size = self.config.vocab_size
                next_scores = beam_scores.unsqueeze(-1) + next_probs
                next_scores = next_scores.view(-1)
                top_k_scores, top_k_tokens = next_scores.topk(beam_width, dim=0)
                
                length_penalty = ((step + 1 + initial_tokens.shape[1]) / (initial_tokens.shape[1])) ** 0.6  # Adjusted penalty
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
         "The tallest mountain in the world is Mount Everest, standing at 8,848 meters above sea level. It is located in the Himalayas on the border between Nepal and the Tibet Autonomous Region of China, a region known for its rugged terrain and extreme weather."),
        ("How do bees contribute to the ecosystem, and why are they considered vital?", 
         "Bees are vital pollinators, transferring pollen between flowers to enable plant reproduction. They support the growth of fruits, vegetables, and nuts, sustaining biodiversity and food chains. Without them, many ecosystems would collapse, affecting agriculture and wildlife significantly."),
        ("What causes the Northern Lights, and where can they be best observed from Earth?", 
         "The Northern Lights are caused by charged particles from the sun hitting gases in Earth's atmosphere, creating colorful displays. They are best observed in high-latitude regions like Norway, Iceland, and Canada, especially during winter months when nights are long and dark."),
        ("Why do leaves change color in the fall, and what chemical processes are involved?", 
         "Leaves change color in fall due to the breakdown of chlorophyll, revealing pigments like carotenoids and anthocyanins. As days shorten, trees stop producing chlorophyll, and cooler temperatures trigger these pigments to dominate, painting leaves in shades of yellow, orange, and red."),
        ("What is the largest desert on Earth, and what unique features does it have?", 
         "The largest desert on Earth is the Antarctic Desert, covering 14 million square kilometers. Unlike hot deserts, it’s a cold desert with vast ice sheets, minimal precipitation, and unique wildlife like penguins and seals adapted to its frigid environment."),
        ("How do volcanoes form, and what role do they play in shaping the planet?", 
         "Volcanoes form when molten rock from Earth’s mantle rises through the crust, often at tectonic plate boundaries. They shape the planet by creating landforms, releasing gases that influence climate, and enriching soil with minerals over time."),
        ("What is the primary source of energy for Earth, and how does it affect climate?", 
         "The primary source of energy for Earth is the Sun, driving weather patterns through solar radiation. It heats the atmosphere unevenly, causing wind, precipitation, and temperature shifts that define global climate systems and influence seasonal changes."),
        ("Why do some animals migrate, and what challenges do they face during migration?", 
         "Some animals migrate to find food, breed, or escape harsh weather, like birds flying south in winter. Challenges include predation, exhaustion, and habitat loss, testing their endurance and navigation skills over vast distances."),
        ("What is the deepest part of the ocean, and what life forms exist there?", 
         "The deepest part of the ocean is the Mariana Trench, reaching 11,000 meters below sea level. Life forms there include extremophiles like tube worms and anglerfish, adapted to high pressure, darkness, and scarce food resources."),
        ("How do hurricanes form, and what conditions are necessary for their development?", 
         "Hurricanes form over warm ocean waters when moist air rises, creating low pressure and spinning winds. They need temperatures above 26.5°C, high humidity, and low wind shear to develop into powerful storms with destructive potential."),
        ("What is the largest mammal on Earth, and what are its key characteristics?", 
         "The largest mammal on Earth is the blue whale, growing up to 30 meters long and weighing 200 tons. It has a streamlined body, baleen plates for filter-feeding krill, and a massive heart, thriving in oceans worldwide."),
        ("Why do stars twinkle, and what does this phenomenon tell us about the atmosphere?", 
         "Stars twinkle because their light bends as it passes through layers of Earth’s turbulent atmosphere. This indicates atmospheric density and temperature variations, revealing conditions that affect visibility and weather patterns on the planet."),
        ("What causes earthquakes, and how are they measured for intensity and impact?", 
         "Earthquakes occur when tectonic plates shift, releasing energy as seismic waves. They’re measured using the Richter scale for magnitude and the Mercalli scale for intensity, assessing damage and ground shaking experienced in affected areas."),
        ("How do coral reefs support marine life, and why are they under threat today?", 
         "Coral reefs provide habitats, breeding grounds, and food for countless marine species, acting as ocean nurseries. They’re threatened by climate change, ocean acidification, and pollution, which cause bleaching and ecosystem collapse."),
        ("What is the primary function of the human heart, and how does it operate?", 
         "The human heart pumps blood throughout the body, delivering oxygen and nutrients. It operates via rhythmic contractions, with four chambers working in sync, regulated by electrical impulses to maintain circulation and support life."),
        ("Why do some birds sing, and what purposes do their songs serve in nature?", 
         "Some birds sing to attract mates, defend territory, or communicate with their flock. Songs signal health and strength, helping reproduction and survival, while also coordinating group behaviors in the wild."),
        ("What is the largest river by volume, and where does it flow through geographically?", 
         "The largest river by volume is the Amazon, discharging 209,000 cubic meters per second. It flows through South America, primarily Brazil and Peru, draining a vast rainforest basin into the Atlantic Ocean."),
        ("How do glaciers form, and what role do they play in Earth’s climate system?", 
         "Glaciers form when snow accumulates and compresses into ice over centuries, typically in cold regions. They regulate sea levels, reflect sunlight to cool the planet, and store freshwater, influencing global climate patterns."),
        ("What causes the tides, and how do they affect coastal ecosystems and human activity?", 
         "Tides are caused by the gravitational pull of the Moon and Sun on Earth’s oceans. They shape coastal ecosystems by nourishing wetlands and influence human activities like fishing, shipping, and tidal energy generation."),
        ("Why do some plants have thorns, and how do these structures benefit their survival?", 
         "Some plants have thorns to deter herbivores, protecting leaves and stems from being eaten. These structures enhance survival by reducing grazing pressure, allowing plants to thrive in harsh or competitive environments."),
        ("What is the hottest planet in our solar system, and why is it so extreme?", 
         "The hottest planet is Venus, with surface temperatures reaching 475°C due to a thick atmosphere trapping heat via the greenhouse effect. Its proximity to the Sun and lack of water exacerbate these extreme conditions."),
        ("How do bats use echolocation, and what advantages does it provide them?", 
         "Bats use echolocation by emitting sound waves that bounce off objects, helping them navigate and hunt in darkness. This provides advantages like precise prey detection and obstacle avoidance, critical for nocturnal survival."),
        ("What is the largest forest in the world, and what makes it ecologically significant?", 
         "The largest forest is the Amazon rainforest, spanning 5.5 million square kilometers. It’s significant for its biodiversity, carbon storage, and oxygen production, acting as a global climate regulator and habitat for countless species."),
        ("Why do some fish live in schools, and what benefits do they gain from this behavior?", 
         "Some fish live in schools to confuse predators, improve foraging, and enhance mating opportunities. This behavior increases survival rates by offering safety in numbers and efficient resource use in their aquatic environments."),
        ("What causes rainbows, and how do they form in the sky after rain?", 
         "Rainbows form when sunlight refracts, reflects, and disperses in water droplets, splitting into colors. They appear after rain when the sun shines through lingering moisture, creating a spectrum visible to the human eye."),
        ("How do penguins survive in cold climates, and what adaptations help them thrive?", 
         "Penguins survive cold climates with thick blubber, dense feathers, and a countercurrent heat exchange system in their flippers. These adaptations insulate them, reduce heat loss, and enable swimming in icy waters."),
        ("What is the primary gas in Earth’s atmosphere, and how does it support life?", 
         "The primary gas in Earth’s atmosphere is nitrogen, making up 78%. It supports life by forming proteins and DNA in organisms, while diluting oxygen to safe levels for respiration and stabilizing atmospheric pressure."),
        ("Why do some animals hibernate, and what physiological changes occur during hibernation?", 
         "Some animals hibernate to conserve energy during food-scarce winters, lowering metabolism and body temperature. Physiological changes include slowed heart rate, reduced breathing, and reliance on stored fat, ensuring survival until spring."),
        ("What is the largest lake by surface area, and where is it located geographically?", 
         "The largest lake by surface area is the Caspian Sea, covering 371,000 square kilometers. It’s located between Europe and Asia, bordered by Russia, Kazakhstan, Turkmenistan, Iran, and Azerbaijan, technically a saltwater lake."),
        ("How do wind patterns affect weather, and what drives these atmospheric movements?", 
         "Wind patterns distribute heat and moisture, shaping weather like storms and droughts. They’re driven by the Sun’s uneven heating of Earth, Earth’s rotation, and pressure differences, creating global circulation systems."),
        ("What causes lightning, and how does it interact with the environment during a storm?", 
         "Lightning occurs when electric charges build up in clouds and discharge to the ground or between clouds. It heats the air, causing thunder, and can ignite fires or enrich soil with nitrogen in the environment."),
        ("Why do some insects have camouflage, and how does it aid their survival in nature?", 
         "Some insects have camouflage to blend into their surroundings, avoiding predators or ambushing prey. This aids survival by enhancing hunting success and reducing detection, critical for their life cycles in diverse habitats."),
        ("What is the brightest star visible from Earth, and what are its notable features?", 
         "The brightest star visible from Earth is Sirius, in the Canis Major constellation. It’s a binary system, 25 times more luminous than the Sun, with a white dwarf companion, located 8.6 light-years away."),
        ("How do rivers shape landscapes, and what geological processes are involved over time?", 
         "Rivers shape landscapes by eroding rock, depositing sediment, and carving valleys over millennia. Geological processes like weathering, erosion, and sediment transport create features like canyons, deltas, and floodplains, altering terrain significantly."),
        ("What is the primary source of freshwater, and how is it replenished naturally?", 
         "The primary source of freshwater is precipitation, like rain and snow, stored in rivers, lakes, and aquifers. It’s replenished by the water cycle, where evaporation from oceans forms clouds that release moisture over land."),
        ("Why do some mammals have fur, and what functions does it serve in their lives?", 
         "Some mammals have fur for insulation, camouflage, and protection from weather and predators. It regulates temperature, aids in sensory perception, and can signal maturity or health, supporting survival across varied environments."),
        ("What causes the seasons, and how do they affect ecosystems around the world?", 
         "Seasons are caused by Earth’s tilted axis as it orbits the Sun, altering sunlight angles. They affect ecosystems by driving plant growth cycles, animal migration, and breeding patterns, shaping biodiversity globally."),
        ("How do clouds form, and what role do they play in weather prediction and climate?", 
         "Clouds form when water vapor cools and condenses around particles in the atmosphere, becoming visible droplets. They indicate weather changes, like rain or storms, and regulate climate by reflecting sunlight or trapping heat."),
        ("What is the largest bird by wingspan, and where can it be found in the wild?", 
         "The largest bird by wingspan is the wandering albatross, reaching up to 3.5 meters. It’s found in the Southern Ocean, soaring over open waters near Antarctica, South America, and Australia, feeding on marine life."),
        ("Why do some trees lose leaves in winter, and how does this adaptation benefit them?", 
         "Some trees lose leaves in winter to conserve water and energy during cold, dry months. This deciduous adaptation reduces frost damage and nutrient loss, allowing survival until spring when conditions improve for growth."),
    ]

def main():
    config = ModelConfig()
    config.update(d_model=512, n_layers=12, vocab_size=1024, max_seq_len=128)

    qa_corpus = get_default_qa_corpus()
    logger.info(f"Corpus size: {len(qa_corpus)} QA pairs")

    logger.info("Initializing model...")
    model = AdvancedLLM(config)

    logger.info("Starting training...")
    model.fit(qa_corpus, epochs=30, learning_rate=3e-4, batch_size=16)

    prompts = [
        "What is the tallest mountain in the world, and where is it located geographically speaking?",
        "How do bees contribute to the ecosystem, and why are they considered vital?",
        "What causes the Northern Lights, and where can they be best observed from Earth?",
    ]

    logger.info("\nTesting the model:")
    for prompt in prompts:
        response = model.generate(prompt, max_length=50, beam_width=config.beam_width)
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Response: {response}\n")

if __name__ == "__main__":
    main()
