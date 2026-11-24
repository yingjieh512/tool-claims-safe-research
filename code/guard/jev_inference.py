"""Standalone typed Jev v3.2 inference. No training imports or generated answers."""
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

SAFETY = ('unsafe', 'safe')


class JevClassifier(nn.Module):
    def __init__(self, checkpoint, base_model='Qwen/Qwen3.5-2B', device='cuda:0', max_length=16384):
        super().__init__(); checkpoint = Path(checkpoint)
        self.device_name = device; self.max_length = max_length
        self.processor = AutoProcessor.from_pretrained(base_model)
        self.tokenizer = self.processor.tokenizer
        dtype = torch.bfloat16 if str(device).startswith('cuda') else torch.float32
        self.full = AutoModelForImageTextToText.from_pretrained(base_model, dtype=dtype,
            attn_implementation='sdpa', device_map={'': device})
        self.full.model.language_model = PeftModel.from_pretrained(
            self.full.model.language_model, checkpoint/'adapter', is_trainable=False)
        self.full.requires_grad_(False); self.full.config.use_cache = False
        hidden = self.full.config.text_config.hidden_size
        self.safety_head = nn.Linear(hidden, 2, device=device, dtype=torch.float32)
        self.category_projection = nn.Linear(hidden, hidden, device=device, dtype=torch.float32)
        heads = torch.load(checkpoint/'heads.pt', map_location=device, weights_only=True)
        self.safety_head.load_state_dict(heads['safety_head'])
        self.category_projection.load_state_dict(heads['category_projection'])
        self.log_scale = nn.Parameter(heads['log_scale'].to(device))
        scores = torch.load(checkpoint/'score-head.pt', map_location=device, weights_only=True)
        self.score_head = nn.Linear(hidden, 1, device=device, dtype=torch.float32)
        self.score_head.load_state_dict(scores['head'])
        self.score_threshold_base = nn.Parameter(scores['base'].to(device))
        self.score_threshold_deltas = nn.Parameter(scores['deltas'].to(device))
        self.label_vectors = {}; self.eval()

    def label_vector(self, label):
        if label not in self.label_vectors:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if not ids: raise ValueError('Category has no tokens')
            embeddings = self.full.get_input_embeddings().weight[ids]
            self.label_vectors[label] = F.normalize(embeddings.float().mean(0), dim=0).detach()
        return self.label_vectors[label]

    @torch.inference_mode()
    def predict(self, text, options, mode='single', safety_applicable=True):
        if not isinstance(text, str) or not text.strip(): raise ValueError('Nonempty text required')
        if mode not in ('single', 'multi', 'score'): raise ValueError('Unsupported mode')
        if len(options) != len(set(options)) or any(not isinstance(x, str) or not x.strip() for x in options):
            raise ValueError('Unique nonempty category strings required')
        if not options and (not safety_applicable or mode != 'single'): raise ValueError('Empty categories require safety-only mode')
        if mode == 'score' and options != ['low', 'medium', 'high', 'critical']: raise ValueError('Unsupported ordinal scale')
        encoded = self.processor.apply_chat_template([{'role':'user','content':text}], tokenize=True,
            return_dict=True, return_tensors='pt', add_generation_prompt=True, enable_thinking=False)
        tokens = encoded['input_ids'].shape[1]
        if tokens > self.max_length: raise ValueError('Context exceeds limit; no truncation performed')
        encoded = {k:v.to(self.device_name) for k,v in encoded.items()}
        result = self.full.model(**encoded, use_cache=False, return_dict=True)
        final = encoded['attention_mask'].sum(-1)-1
        hidden = result.last_hidden_state[torch.arange(final.numel(), device=final.device), final].float()
        sp = self.safety_head(hidden)[0].softmax(-1).float().cpu().tolist()
        score = None
        if mode == 'score':
            thresholds = torch.cat([self.score_threshold_base.reshape(1), self.score_threshold_base+
                F.softplus(self.score_threshold_deltas).cumsum(0)])
            q = (self.score_head(hidden)[0]-thresholds).sigmoid().float()
            probabilities = torch.cat([1-q[:1], q[:-1]-q[1:], q[-1:]]).cpu().tolist()
            score = float(q.sum()/3)
        elif options:
            query = F.normalize(self.category_projection(hidden), dim=-1)
            vectors = torch.stack([self.label_vector(x) for x in options])
            logits = (query @ vectors.T * self.log_scale.exp().clamp(max=100.))[0]
            probabilities = (logits.sigmoid() if mode == 'multi' else logits.softmax(-1)).float().cpu().tolist()
        else: probabilities = []
        chosen = ([x for x,p in zip(options,probabilities) if p>=.5] if mode=='multi' else
            [options[max(range(len(options)),key=probabilities.__getitem__)]] if options else [])
        return {'model':'jev-v3.2-2b','mode':mode,'probabilities':dict(zip(options,probabilities)),
            'categories':chosen,'safety':SAFETY[max(range(2),key=sp.__getitem__)] if safety_applicable else None,
            'safety_probs':dict(zip(SAFETY,sp)) if safety_applicable else None,
            'score':score,'prompt_tokens':tokens,'truncated':False}
