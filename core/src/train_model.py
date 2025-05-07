import os
import sys
import logging
import json
import time
import math
import warnings
import random

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import logging as transformers_logging
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from config.config import CONFIG
from tqdm.auto import tqdm

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ----- Model Definition -----
class LCF_ATEPC(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased',
                 hidden_size=768, num_aspect_labels=2, num_sentiment_labels=2,
                 context_window=3, dropout_rate=0.15):
        super(LCF_ATEPC, self).__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.hidden_size = hidden_size
        self.context_window = context_window
        
        # Improved dropout
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        
        # Layer normalization for better stability
        self.ate_layer_norm = nn.LayerNorm(hidden_size)
        self.apc_layer_norm = nn.LayerNorm(hidden_size)
        
        # Self-attention mechanisms for aspect-based context modeling
        self.aspect_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Fusion layer for context integration
        self.fusion_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.fusion_act = nn.GELU()
        
        # Output classifiers with separate dropout
        self.aspect_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, num_aspect_labels)
        )
        
        self.sentiment_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, num_sentiment_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, aspect_positions=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=True)
        sequence_output = outputs.last_hidden_state
        
        # Extract features for Aspect Term Extraction
        normalized_seq_output = self.ate_layer_norm(sequence_output)
        ate_logits = self.aspect_classifier(normalized_seq_output)

        apc_logits = None
        if aspect_positions is not None:
            batch_size, seq_len, hid = sequence_output.size()
            apc_feats = []

            actual_batch_size = min(batch_size, len(aspect_positions))
            
            for b in range(actual_batch_size):
                try:
                    spans = aspect_positions[b]
                    
                    if not spans:
                        # If no aspect spans, use CLS token
                        apc_feats.append(sequence_output[b, 0])
                        continue
                        
                    # Normalize span format
                    if not isinstance(spans, list) and not isinstance(spans, tuple):
                        spans = [spans]
                    elif len(spans) == 2 and all(isinstance(x, (int, float)) for x in spans):
                        spans = [spans]
                    
                    batch_span_feats = []
                    for span in spans:
                        if isinstance(span, (list, tuple)) and len(span) == 2:
                            s, e = span
                            s = max(0, min(int(s), seq_len - 1))
                            e = max(s, min(int(e), seq_len - 1))
                            
                            # Get local context around aspect
                            left = max(0, s - self.context_window)
                            right = min(seq_len, e + 1 + self.context_window)
                            
                            # Extract aspect representation
                            aspect_repr = sequence_output[b, s:e, :]
                            if aspect_repr.size(0) == 0:  # Empty aspect
                                aspect_repr = sequence_output[b, 0].unsqueeze(0)  # Use CLS
                            else:
                                aspect_repr = aspect_repr.mean(dim=0, keepdim=True)
                                
                            # Extract context representation
                            context_repr = sequence_output[b, left:right, :]
                            
                            # Apply self-attention to enhance aspect-context relationship
                            if context_repr.size(0) > 1:  # Need at least 2 tokens for attention
                                context_mask = torch.ones(context_repr.size(0), device=context_repr.device)
                                context_attn_output, _ = self.aspect_attention(
                                    context_repr.unsqueeze(0),
                                    context_repr.unsqueeze(0),
                                    context_repr.unsqueeze(0),
                                    key_padding_mask=(1 - context_mask.unsqueeze(0)).bool()
                                )
                                context_repr = context_attn_output.squeeze(0)
                                context_repr = context_repr.mean(dim=0, keepdim=True)
                            else:
                                context_repr = context_repr.mean(dim=0, keepdim=True)
                            
                            # Combine aspect and context
                            combined = torch.cat([aspect_repr, context_repr], dim=1)
                            fused = self.fusion_fc(combined.view(-1))
                            fused = self.fusion_act(fused)
                            
                            batch_span_feats.append(fused)
                    
                    # Combine features from multiple aspects if present
                    if not batch_span_feats:
                        apc_feats.append(sequence_output[b, 0])  # Fallback to CLS
                    else:
                        span_tensor = torch.stack(batch_span_feats)
                        apc_feats.append(span_tensor.mean(dim=0))
                        
                except Exception as e:
                    # Fallback to CLS token if any error occurs
                    apc_feats.append(sequence_output[b, 0])
            
            # Handle case when batch is smaller than expected
            if actual_batch_size < batch_size:
                for b in range(actual_batch_size, batch_size):
                    apc_feats.append(sequence_output[b, 0])
            
            # Stack all features and normalize
            apc_tensor = torch.stack(apc_feats, dim=0)
            apc_tensor = self.apc_layer_norm(apc_tensor)
            
            # Apply sentiment classifier
            apc_logits = self.sentiment_classifier(apc_tensor)

        return ate_logits, apc_logits

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.bert, "gradient_checkpointing_enable"):
            self.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        else:
            if gradient_checkpointing_kwargs is not None:
                logger.warning("Gradient checkpointing kwargs not supported for older transformer versions. Ignoring.")
            if hasattr(self.bert, "config") and hasattr(self.bert.config, "use_cache"):
                self.bert.config.use_cache = False
            self.bert.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        if hasattr(self.bert, "gradient_checkpointing_disable"):
            self.bert.gradient_checkpointing_disable()
        else:
            if hasattr(self.bert, "config") and hasattr(self.bert.config, "use_cache"):
                self.bert.config.use_cache = True
            self.bert.gradient_checkpointing = False
            
    def freeze_bert_layers(self, num_layers_to_freeze=None):
        """
        Freeze BERT layers for transfer learning
        
        Args:
            num_layers_to_freeze: Number of encoder layers to freeze from bottom.
                                If None, freeze all BERT parameters.
        """
        if num_layers_to_freeze is None:
            # Freeze all BERT parameters
            for param in self.bert.parameters():
                param.requires_grad = False
            logger.info("Froze all BERT parameters")
            return
            
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
            
        # Freeze specific encoder layers
        if hasattr(self.bert, "encoder") and hasattr(self.bert.encoder, "layer"):
            layers = self.bert.encoder.layer
            for i, layer in enumerate(layers):
                if i < num_layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
            
            logger.info(f"Froze embeddings and {num_layers_to_freeze} encoder layers")

# ----- Dataset & Preprocessing -----
class ABSADataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer_name="bert-base-uncased", max_length=128, inference=False, augment=False):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.inference = inference
        self.augment = augment and not inference
        self.sentiment2idx = {'negative': 0, 'positive': 1}

        self.data = dataframe.copy()
        required_cols = ["review", "aspect"]
        if not all(col in self.data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.data.columns]
            raise ValueError(f"Missing required columns: {missing}")
            
        self.data["review"] = self.data["review"].astype(str)
        self.data["aspect"] = self.data["aspect"].astype(str)

        if not inference and "sentiment" in self.data.columns:
            if self.data["sentiment"].dtype == object:
                self.data["sentiment"] = self.data["sentiment"].apply(lambda x: self.sentiment2idx.get(x, x) if isinstance(x, str) else x)
            try:
                self.data["sentiment"] = self.data["sentiment"].astype(int)
            except Exception as e:
                logger.warning(f"Error converting labels to int: {e}")
                logger.warning("Creating default labels (0) for problematic rows")
                self.data["sentiment"] = 0
                
        # Load stopwords for data augmentation
        try:
            import nltk
            from nltk.corpus import stopwords
            try:
                self.stopwords = set(stopwords.words('english'))
            except:
                nltk.download('stopwords', quiet=True)
                self.stopwords = set(stopwords.words('english'))
        except:
            self.stopwords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
                                  'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                                  'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
                                  'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                                  'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                                  'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                                  'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                                  'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
                                  'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                                  'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
                                  'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
                                  'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
                                  'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
                                  't', 'can', 'will', 'just', 'don', 'should', 'now'])

    def __len__(self):
        return len(self.data)
    
    def _augment_text(self, text, aspect):
        """Simple data augmentation techniques"""
        
        # Don't augment very short texts
        if len(text.split()) < 10:
            return text
            
        # Keep aspect untouched
        if aspect.lower() not in text.lower():
            return text
            
        aspect_lower = aspect.lower()
        words = text.split()
        augmented_words = []
        
        # 1. Random word dropout (except aspect)
        if random.random() < 0.5:
            for word in words:
                if (word.lower() not in aspect_lower and 
                    word.lower() in self.stopwords and 
                    random.random() < 0.15):
                    continue  # Drop this word
                augmented_words.append(word)
            return " ".join(augmented_words) if augmented_words else text
            
        # 2. Random word swap (except aspect)
        elif random.random() < 0.5:
            n = len(words)
            if n > 4:  # Only for longer texts
                for _ in range(min(3, n // 5)):  # Swap up to 3 pairs
                    i, j = random.sample(range(n), 2)
                    # Don't swap words in the aspect
                    if (words[i].lower() not in aspect_lower and 
                        words[j].lower() not in aspect_lower):
                        words[i], words[j] = words[j], words[i]
            return " ".join(words)
            
        return text

    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            text = str(row["review"])
            aspect = str(row["aspect"])
            
            # Apply data augmentation if enabled
            if self.augment and random.random() < 0.3:  # 30% chance of augmentation
                text = self._augment_text(text, aspect)
            
            marked_text = text
            aspect_lower = aspect.lower().strip()
            review_lower = text.lower()
            
            if aspect_lower and aspect_lower in review_lower:
                aspect_start = review_lower.find(aspect_lower)
                aspect_end = aspect_start + len(aspect_lower)
                marked_text = f"{text[:aspect_start]}[ASPECT]{text[aspect_start:aspect_end]}[/ASPECT]{text[aspect_end:]}"
            
            encoding = self.tokenizer(
                marked_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            aspect_indices = [-1, -1]
            input_ids = encoding["input_ids"].squeeze().tolist()
            aspect_token_ids = self.tokenizer.encode("[ASPECT]", add_special_tokens=False)
            end_aspect_token_ids = self.tokenizer.encode("[/ASPECT]", add_special_tokens=False)
            
            try:
                for i in range(len(input_ids)):
                    if i < len(input_ids) - len(aspect_token_ids) and input_ids[i:i+len(aspect_token_ids)] == aspect_token_ids:
                        aspect_indices[0] = i + len(aspect_token_ids)
                    if i < len(input_ids) - len(end_aspect_token_ids) and input_ids[i:i+len(end_aspect_token_ids)] == end_aspect_token_ids:
                        aspect_indices[1] = i
                        break
                        
                if aspect_indices[0] == -1 or aspect_indices[1] == -1 or aspect_indices[0] >= aspect_indices[1]:
                    special_tokens = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id}
                    for i, token_id in enumerate(input_ids):
                        if token_id not in special_tokens:
                            aspect_indices[0] = i
                            break
                    
                    if aspect_indices[0] != -1:
                        non_pad_length = sum(1 for x in input_ids if x != self.tokenizer.pad_token_id)
                        aspect_length = max(int(non_pad_length * 0.2), 1)
                        aspect_indices[1] = min(aspect_indices[0] + aspect_length, len(input_ids) - 1)
                
                if aspect_indices[0] < 0:
                    aspect_indices[0] = 1
                if aspect_indices[1] <= aspect_indices[0]:
                    aspect_indices[1] = min(aspect_indices[0] + 1, self.max_length - 1)
                
            except Exception as e:
                logger.warning(f"Error finding aspect markers: {e}")
                aspect_indices = [1, min(5, self.max_length - 1)]
            
            for key in encoding:
                if isinstance(encoding[key], torch.Tensor) and encoding[key].dim() == 2:
                    encoding[key] = encoding[key].squeeze(0)
            
            aspect_labels = torch.zeros(self.max_length, dtype=torch.long)
            if aspect_indices[0] != -1 and aspect_indices[1] != -1 and aspect_indices[0] < aspect_indices[1]:
                aspect_labels[aspect_indices[0]:aspect_indices[1]] = 1
            
            item = {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "token_type_ids": encoding["token_type_ids"],
                "aspect_positions": [[aspect_indices[0], aspect_indices[1]]],
                "aspect_labels": aspect_labels,
                "sentiment_label": torch.tensor(0, dtype=torch.long)
            }
            
            if not self.inference and "sentiment" in row:
                try:
                    label_value = int(row["sentiment"])
                    if label_value == 2:
                        label_value = 1
                    item["sentiment_label"] = torch.tensor(label_value, dtype=torch.long)
                except:
                    item["sentiment_label"] = torch.tensor(0, dtype=torch.long)
                
            return item
            
        except Exception as e:
            logger.error(f"Critical error processing sample {idx}: {e}")
            dummy_input = torch.ones(self.max_length, dtype=torch.long)
            dummy_mask = torch.ones(self.max_length, dtype=torch.long)
            
            dummy_item = {
                "input_ids": dummy_input,
                "attention_mask": dummy_mask,
                "token_type_ids": torch.zeros(self.max_length, dtype=torch.long),
                "aspect_positions": [[1, 2]],
                "aspect_labels": torch.zeros(self.max_length, dtype=torch.long),
                "sentiment_label": torch.tensor(0, dtype=torch.long)
            }
            
            return dummy_item

# Dynamic batch size scheduler
class BatchSizeScheduler:
    def __init__(self, initial_batch_size, max_batch_size, factor=1.25, epochs_to_increase=2):
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.factor = factor  # How much to increase each time
        self.epochs_to_increase = epochs_to_increase  # How many epochs before increasing
        self.current_batch_size = initial_batch_size
        self.epoch = 0
        
    def step(self):
        """Call at the end of each epoch to potentially update batch size"""
        self.epoch += 1
        if self.epoch % self.epochs_to_increase == 0:
            new_batch_size = min(int(self.current_batch_size * self.factor), self.max_batch_size)
            if new_batch_size > self.current_batch_size:
                self.current_batch_size = new_batch_size
                return True  # Batch size changed
        return False  # No change
        
    def get_batch_size(self):
        """Get the current batch size"""
        return self.current_batch_size

# ----- Training & Evaluation -----
def train_epoch(model, dataloader, optimizer, scheduler, device, gradient_accumulation_steps=1, use_amp=True):
    model.train()
    total_loss = 0
    ate_loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    apc_loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    
    # Setup for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device) if batch['token_type_ids'] is not None else None
        aspect_positions = batch['aspect_positions']
        aspect_labels = batch['aspect_labels'].to(device)
        sentiment_label = batch['sentiment_label'].to(device)
        
        # Mixed precision forward pass
        if use_amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                ate_logits, apc_logits = model(
                    input_ids, attention_mask, token_type_ids, aspect_positions
                )
                ate_loss = ate_loss_fn(
                    ate_logits.view(-1, ate_logits.size(-1)),
                    aspect_labels.view(-1)
                )
                apc_loss = apc_loss_fn(apc_logits, sentiment_label)
                loss = ate_loss + apc_loss
                loss = loss / gradient_accumulation_steps
                
            # Mixed precision backward
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(dataloader) - 1:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            # Standard precision forward/backward
            ate_logits, apc_logits = model(
                input_ids, attention_mask, token_type_ids, aspect_positions
            )
            ate_loss = ate_loss_fn(
                ate_logits.view(-1, ate_logits.size(-1)),
                aspect_labels.view(-1)
            )
            apc_loss = apc_loss_fn(apc_logits, sentiment_label)
            loss = ate_loss + apc_loss
            loss = loss / gradient_accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
        total_loss += loss.item() * gradient_accumulation_steps

    return total_loss / len(dataloader)

# Focal Loss implementation for better handling of class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, logits, targets):
        ce_loss = self.ce_loss_fn(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

@torch.no_grad()
def eval_epoch(model, dataloader, device, use_amp=True):
    model.eval()
    total_loss = 0
    correct_ate = 0
    total_tokens = 0
    
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    tn = 0  # True Negatives

    ate_loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    apc_loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device) if batch['token_type_ids'] is not None else None
        aspect_positions = batch['aspect_positions']
        aspect_labels = batch['aspect_labels'].to(device)
        sentiment_label = batch['sentiment_label'].to(device)

        # Mixed precision inference
        if use_amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                ate_logits, apc_logits = model(
                    input_ids, attention_mask, token_type_ids, aspect_positions
                )
                ate_loss = ate_loss_fn(
                    ate_logits.view(-1, ate_logits.size(-1)),
                    aspect_labels.view(-1)
                )
                apc_loss = apc_loss_fn(apc_logits, sentiment_label)
                loss = ate_loss + apc_loss
        else:
            ate_logits, apc_logits = model(
                input_ids, attention_mask, token_type_ids, aspect_positions
            )
            ate_loss = ate_loss_fn(
                ate_logits.view(-1, ate_logits.size(-1)),
                aspect_labels.view(-1)
            )
            apc_loss = apc_loss_fn(apc_logits, sentiment_label)
            loss = ate_loss + apc_loss
            
        total_loss += loss.item()

        ate_preds = ate_logits.argmax(-1)
        mask = attention_mask.bool().view(-1)
        correct_ate += (ate_preds.view(-1)[mask] == aspect_labels.view(-1)[mask]).sum().item()
        total_tokens += mask.sum().item()

        apc_preds = apc_logits.argmax(-1)
        
        for pred, true in zip(apc_preds.cpu().numpy(), sentiment_label.cpu().numpy()):
            if pred == 1 and true == 1:
                tp += 1
            elif pred == 1 and true == 0:
                fp += 1
            elif pred == 0 and true == 1:
                fn += 1
            else:  # pred == 0 and true == 0
                tn += 1

    correct_apc = tp + tn
    accuracy = correct_apc / (tp + tn + fp + fn)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'loss': total_loss / len(dataloader),
        'ate_acc': correct_ate / total_tokens,
        'apc_acc': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def custom_collate_fn(batch):
    """
    Custom function for batch collation.
    Ensures proper handling of aspect_positions.
    """
    elem = batch[0]
    batch_dict = {}
    
    for key in elem:
        if key == "aspect_positions":
            batch_dict[key] = [d[key] for d in batch]
        elif isinstance(elem[key], torch.Tensor):
            batch_dict[key] = torch.stack([d[key] for d in batch])
        else:
            batch_dict[key] = [d[key] for d in batch]
            
    return batch_dict

def save_model_for_deployment(model_path, tokenizer=None, model_config=None):
    """
    Save the model, tokenizer and configuration for deployment in production environments.
    
    This function ensures all necessary components for inference are saved together:
    1. The model weights (already saved by train_model)
    2. The tokenizer for text preprocessing
    3. Model configuration for reconstruction
    4. Inference helper functions and scripts
    
    Args:
        model_path (str): Path to the directory where the model is saved
        tokenizer (transformers.PreTrainedTokenizer, optional): Tokenizer to save
        model_config (dict, optional): Model configuration dictionary
        
    Returns:
        None
    
    Raises:
        FileNotFoundError: If model_path does not exist
        Exception: If there is an error saving components
    """
    import shutil
    import json
    from transformers import BertTokenizer
    
    logger.info(f"Saving model deployment assets to {model_path}")
    
    try:
        os.makedirs(model_path, exist_ok=True)
        
        # 1. Save tokenizer if provided, otherwise use default
        if tokenizer is None:
            logger.info("No tokenizer provided, saving default BERT tokenizer")
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        tokenizer_path = os.path.join(model_path, "tokenizer")
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_path)
        logger.info(f"Tokenizer saved to {tokenizer_path}")
        
        # 2. Save inference configuration
        if model_config is None:
            model_config = {
                "model_type": "LCF_ATEPC",
                "pretrained_model_name": "bert-base-uncased",
                "num_aspect_labels": 2,
                "num_sentiment_labels": 2,
                "context_window": 3,
                "max_seq_length": 128
            }
        
        # Add inference-specific parameters
        inference_config = {
            **model_config,
            "version": "1.0.0",
            "aspect_preprocessing": {
                "use_aspect_markers": True,
                "marker_start": "[ASPECT]",
                "marker_end": "[/ASPECT]"
            },
            "sentiment_mapping": {
                "0": "negative",
                "1": "positive"
            }
        }
        
        with open(os.path.join(model_path, "inference_config.json"), "w") as f:
            json.dump(inference_config, f, indent=2)
        logger.info(f"Inference configuration saved")
        
        # 3. Create a simple inference script
        inference_code = """
import os
import torch
import torch.nn as nn
import json
from transformers import BertTokenizer, BertModel

class LCF_ATEPC(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased',
                 hidden_size=768, num_aspect_labels=2, num_sentiment_labels=2,
                 context_window=3, dropout_rate=0.15):
        super(LCF_ATEPC, self).__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.hidden_size = hidden_size
        self.context_window = context_window
        
        # Improved dropout
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        
        # Layer normalization for better stability
        self.ate_layer_norm = nn.LayerNorm(hidden_size)
        self.apc_layer_norm = nn.LayerNorm(hidden_size)
        
        # Self-attention mechanisms for aspect-based context modeling
        self.aspect_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Fusion layer for context integration
        self.fusion_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.fusion_act = nn.GELU()
        
        # Output classifiers with separate dropout
        self.aspect_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, num_aspect_labels)
        )
        
        self.sentiment_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, num_sentiment_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, aspect_positions=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=True)
        sequence_output = outputs.last_hidden_state
        
        # For inference we only need sentiment prediction
        apc_logits = None
        if aspect_positions is not None:
            batch_size, seq_len, hid = sequence_output.size()
            apc_feats = []

            actual_batch_size = min(batch_size, len(aspect_positions))
            
            for b in range(actual_batch_size):
                try:
                    spans = aspect_positions[b]
                    
                    if not spans:
                        # If no aspect spans, use CLS token
                        apc_feats.append(sequence_output[b, 0])
                        continue
                        
                    # Normalize span format
                    if not isinstance(spans, list) and not isinstance(spans, tuple):
                        spans = [spans]
                    elif len(spans) == 2 and all(isinstance(x, (int, float)) for x in spans):
                        spans = [spans]
                    
                    batch_span_feats = []
                    for span in spans:
                        if isinstance(span, (list, tuple)) and len(span) == 2:
                            s, e = span
                            s = max(0, min(int(s), seq_len - 1))
                            e = max(s, min(int(e), seq_len - 1))
                            
                            # Get local context around aspect
                            left = max(0, s - self.context_window)
                            right = min(seq_len, e + 1 + self.context_window)
                            
                            # Extract aspect representation
                            aspect_repr = sequence_output[b, s:e, :]
                            if aspect_repr.size(0) == 0:  # Empty aspect
                                aspect_repr = sequence_output[b, 0].unsqueeze(0)  # Use CLS
                            else:
                                aspect_repr = aspect_repr.mean(dim=0, keepdim=True)
                                
                            # Extract context representation
                            context_repr = sequence_output[b, left:right, :]
                            
                            # Apply self-attention to enhance aspect-context relationship
                            if context_repr.size(0) > 1:  # Need at least 2 tokens for attention
                                context_mask = torch.ones(context_repr.size(0), device=context_repr.device)
                                context_attn_output, _ = self.aspect_attention(
                                    context_repr.unsqueeze(0),
                                    context_repr.unsqueeze(0),
                                    context_repr.unsqueeze(0),
                                    key_padding_mask=(1 - context_mask.unsqueeze(0)).bool()
                                )
                                context_repr = context_attn_output.squeeze(0)
                                context_repr = context_repr.mean(dim=0, keepdim=True)
                            else:
                                context_repr = context_repr.mean(dim=0, keepdim=True)
                            
                            # Combine aspect and context
                            combined = torch.cat([aspect_repr, context_repr], dim=1)
                            fused = self.fusion_fc(combined.view(-1))
                            fused = self.fusion_act(fused)
                            
                            batch_span_feats.append(fused)
                    
                    # Combine features from multiple aspects if present
                    if not batch_span_feats:
                        apc_feats.append(sequence_output[b, 0])  # Fallback to CLS
                    else:
                        span_tensor = torch.stack(batch_span_feats)
                        apc_feats.append(span_tensor.mean(dim=0))
                        
                except Exception as e:
                    # Fallback to CLS token if any error occurs
                    apc_feats.append(sequence_output[b, 0])
            
            # Handle case when batch is smaller than expected
            if actual_batch_size < batch_size:
                for b in range(actual_batch_size, batch_size):
                    apc_feats.append(sequence_output[b, 0])
            
            # Stack all features and normalize
            apc_tensor = torch.stack(apc_feats, dim=0)
            apc_tensor = self.apc_layer_norm(apc_tensor)
            
            # Apply sentiment classifier
            apc_logits = self.sentiment_classifier(apc_tensor)

        return apc_logits

class AspectSentimentAnalyzer:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(os.path.join(model_path, "inference_config.json"), "r") as f:
            self.config = json.load(f)
        
        # Load tokenizer
        tokenizer_path = os.path.join(model_path, "tokenizer")
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
        # Initialize model
        self.model = LCF_ATEPC(
            pretrained_model_name=self.config["pretrained_model_name"],
            num_aspect_labels=self.config["num_aspect_labels"],
            num_sentiment_labels=self.config["num_sentiment_labels"],
            context_window=self.config["context_window"]
        )
        
        # Load model weights
        model_weights_path = os.path.join(model_path, "model.pt")
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Sentiment mapping
        self.sentiment_map = self.config["sentiment_mapping"]
    
    def analyze(self, text, aspect):
        # Prepare input with aspect markers
        marked_text = text
        aspect_lower = aspect.lower().strip()
        text_lower = text.lower()
        
        if aspect_lower in text_lower:
            start_idx = text_lower.find(aspect_lower)
            end_idx = start_idx + len(aspect_lower)
            marked_text = f"{text[:start_idx]}[ASPECT]{text[start_idx:end_idx]}[/ASPECT]{text[end_idx:]}"
        
        # Tokenize
        encoding = self.tokenizer(
            marked_text,
            padding="max_length",
            truncation=True,
            max_length=self.config.get("max_seq_length", 128),
            return_tensors="pt"
        ).to(self.device)
        
        # Find aspect positions
        input_ids = encoding["input_ids"].squeeze().tolist()
        aspect_indices = [-1, -1]
        aspect_token_ids = self.tokenizer.encode("[ASPECT]", add_special_tokens=False)
        end_aspect_token_ids = self.tokenizer.encode("[/ASPECT]", add_special_tokens=False)
        
        try:
            for i in range(len(input_ids)):
                if i < len(input_ids) - len(aspect_token_ids) and input_ids[i:i+len(aspect_token_ids)] == aspect_token_ids:
                    aspect_indices[0] = i + len(aspect_token_ids)
                if i < len(input_ids) - len(end_aspect_token_ids) and input_ids[i:i+len(end_aspect_token_ids)] == end_aspect_token_ids:
                    aspect_indices[1] = i
                    break
                    
            if aspect_indices[0] == -1 or aspect_indices[1] == -1 or aspect_indices[0] >= aspect_indices[1]:
                # Fallback - estimate position
                special_tokens = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id}
                for i, token_id in enumerate(input_ids):
                    if token_id not in special_tokens:
                        aspect_indices[0] = i
                        break
                
                if aspect_indices[0] != -1:
                    non_pad_length = sum(1 for x in input_ids if x != self.tokenizer.pad_token_id)
                    aspect_length = max(int(non_pad_length * 0.2), 1)
                    aspect_indices[1] = min(aspect_indices[0] + aspect_length, len(input_ids) - 1)
            
            if aspect_indices[0] < 0:
                aspect_indices[0] = 1
            if aspect_indices[1] <= aspect_indices[0]:
                aspect_indices[1] = min(aspect_indices[0] + 1, self.config.get("max_seq_length", 128) - 1)
            
        except Exception:
            aspect_indices = [1, 5]  # Default fallback
        
        # Run inference
        with torch.no_grad():
            logits = self.model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                token_type_ids=encoding["token_type_ids"],
                aspect_positions=[[aspect_indices[0], aspect_indices[1]]]
            )
            
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
            
            sentiment = self.sentiment_map[str(prediction)]
            
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "aspect": aspect,
            "text": text
        }
"""
        
        with open(os.path.join(model_path, "inference.py"), "w") as f:
            f.write(inference_code.strip())
        logger.info("Inference script saved")
        
        # 4. Create a README file with usage instructions
        readme = """# LCF-ATEPC Aspect-Based Sentiment Analysis Model

## Model Overview
This is a fine-tuned LCF-ATEPC (Local Context Focus for Aspect Term Extraction and Polarity Classification) model
for aspect-based sentiment analysis.

## Usage

```python
from inference import AspectSentimentAnalyzer

# Initialize analyzer with model path
analyzer = AspectSentimentAnalyzer("path/to/model_directory")

# Analyze sentiment for a specific aspect
result = analyzer.analyze(
    text="The food was excellent but the service was terrible.",
    aspect="service"
)

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## Model Details
- Base model: BERT
- Task: Aspect-Based Sentiment Analysis
- Input: Text review and aspect term
- Output: Sentiment classification (positive/negative) with confidence

## Requirements
- PyTorch >= 1.10.0
- Transformers >= 4.15.0
"""
        
        with open(os.path.join(model_path, "README.md"), "w") as f:
            f.write(readme)
        logger.info("README file with usage instructions saved")
        
        logger.info("Model successfully prepared for deployment")
        
    except Exception as e:
        logger.error(f"Error preparing model for deployment: {e}")
        raise

def train_model(train_dataset, val_dataset):
    """
    Train the LCF-ATEPC model on the given datasets.
    
    Args:
        train_dataset: Dataset containing training examples
        val_dataset: Dataset containing validation examples
        
    Returns:
        str: Path to the saved best model
    """
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model_name = "bert-base-uncased"
    try:
        model = LCF_ATEPC(pretrained_model_name=model_name, num_aspect_labels=2, num_sentiment_labels=2)
        model = model.to(device)
        
        # Freeze BERT layers initially for transfer learning
        model.freeze_bert_layers(num_layers_to_freeze=10)  # Start by freezing 10 out of 12 BERT layers
        
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise

    output_dir = CONFIG.get("lcf_atepc_output_path", "outputs/lcf-atepc")
    os.makedirs(output_dir, exist_ok=True)

    # Optimized training parameters
    initial_batch_size = 16
    max_batch_size = 48
    eval_batch_size = 64
    gradient_accumulation_steps = 2
    weight_decay = 0.01
    learning_rate = 2e-5
    num_epochs = 15  # Increased epochs with early stopping
    use_amp = True if torch.cuda.is_available() else False
    
    # Initialize batch size scheduler
    batch_scheduler = BatchSizeScheduler(
        initial_batch_size=initial_batch_size, 
        max_batch_size=max_batch_size,
        factor=1.5,
        epochs_to_increase=2
    )
    
    current_batch_size = batch_scheduler.get_batch_size()
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=current_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    # Layer-wise learning rates
    optimizer = AdamW(
        [
            {"params": model.bert.parameters(), "lr": learning_rate},
            {"params": model.aspect_classifier.parameters(), "lr": learning_rate * 5},  # Reduced from 10x to 5x
            {"params": model.sentiment_classifier.parameters(), "lr": learning_rate * 5}  # Reduced from 10x to 5x
        ],
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-8
    )
    
    # Adjust steps for gradient accumulation
    effective_batch_size = current_batch_size * gradient_accumulation_steps
    total_steps = (len(train_loader) * num_epochs) // gradient_accumulation_steps
    warmup_steps = int(0.1 * total_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Enable gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    best_f1 = 0.0
    best_model_path = os.path.join(output_dir, "best_model")
    early_stopping_counter = 0
    early_stopping_patience = 3
    
    # Exponential moving average for metrics
    ema_alpha = 0.2
    ema_f1 = 0.0
    
    # Progressive unfreezing
    unfreeze_stages = [(3, ['classifier']), (5, ['encoder.layer.11']), (7, [])]
    current_unfreeze_idx = 0
    
    if use_amp:
        logger.info("Using mixed precision training (FP16)")
    
    try:
        logger.info("Starting training...")
        
        for epoch in range(1, num_epochs + 1):
            # Progressive unfreezing logic
            if current_unfreeze_idx < len(unfreeze_stages) and epoch >= unfreeze_stages[current_unfreeze_idx][0]:
                stage_epoch, layers_to_unfreeze = unfreeze_stages[current_unfreeze_idx]
                if not layers_to_unfreeze:  # Empty list means unfreeze all
                    for param in model.parameters():
                        param.requires_grad = True
                    logger.info(f"Epoch {epoch}: Unfreezing all layers")
                else:
                    for name, param in model.named_parameters():
                        if any(layer in name for layer in layers_to_unfreeze):
                            param.requires_grad = True
                    logger.info(f"Epoch {epoch}: Unfreezing {layers_to_unfreeze}")
                current_unfreeze_idx += 1
            
            # Check if batch size needs updating
            if batch_scheduler.step() and epoch < num_epochs - 2:  # Don't increase batch in last 2 epochs
                new_batch_size = batch_scheduler.get_batch_size()
                logger.info(f"Increasing batch size from {current_batch_size} to {new_batch_size}")
                
                # Recreate dataloader with new batch size
                current_batch_size = new_batch_size
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=current_batch_size,
                    shuffle=True,
                    num_workers=0,
                    collate_fn=custom_collate_fn
                )
                
                # Adjust gradient accumulation steps if needed
                if current_batch_size >= 32:
                    gradient_accumulation_steps = 1
                    logger.info("Reduced gradient accumulation steps to 1")
            
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, 
                                     gradient_accumulation_steps=gradient_accumulation_steps,
                                     use_amp=use_amp)
            metrics = eval_epoch(model, val_loader, device, use_amp=use_amp)
            
            # Calculate combined score with EMA
            current_f1 = metrics['f1_score'] * 0.6 + metrics['ate_acc'] * 0.4
            if epoch == 1:
                ema_f1 = current_f1
            else:
                ema_f1 = ema_alpha * current_f1 + (1 - ema_alpha) * ema_f1
            
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {metrics['loss']:.4f}")
            logger.info(f"ATE Accuracy: {metrics['ate_acc']:.4f}")
            logger.info(f"APC Accuracy: {metrics['apc_acc']:.4f}")
            logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
            logger.info(f"Combined Score: {current_f1:.4f}")
            logger.info(f"EMA Score: {ema_f1:.4f}")
            logger.info(f"Current batch size: {current_batch_size}")
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                early_stopping_counter = 0
                
                # Save best model
                os.makedirs(best_model_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(best_model_path, "model.pt"))
                
                model_config = {
                    "pretrained_model_name": model_name,
                    "num_aspect_labels": 2,
                    "num_sentiment_labels": 2,
                    "context_window": 3
                }
                
                model_info = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_metrics": metrics,
                    "best_f1": best_f1,
                    "ema_f1": ema_f1,
                    "model_config": model_config,
                    "batch_size": current_batch_size
                }
                
                with open(os.path.join(best_model_path, "model_info.json"), "w") as f:
                    json.dump(model_info, f, indent=2)
                
                logger.info(f"New best model saved with F1: {best_f1:.4f}")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
    
    except Exception as e:
        logger.error(f"Error during training: {e}")
        if best_f1 > 0:
            logger.info("Saving last successful model despite error...")
            os.makedirs(best_model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(best_model_path, "last_model.pt"))
        raise
    
    # Save additional deployment files for the best model
    try:
        logger.info("Preparing model for deployment...")
        save_model_for_deployment(
            best_model_path,
            tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"),
            model_config=model_config
        )
    except Exception as e:
        logger.error(f"Error saving deployment files: {e}")
        logger.info("Training completed successfully but deployment files could not be saved.")
    
    return best_model_path

def prepare_data_format(input_path, output_path):
    """
    Prepare data in the required format for LCF-ATEPC.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to save the processed CSV file
    """
    try:
        df = pd.read_csv(input_path)

        required_columns = ['review', 'aspect', 'sentiment']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if df['sentiment'].dtype == object:
            sentiment_map = {'negative': 0, 'positive': 1}
            df['sentiment'] = df['sentiment'].map(sentiment_map)
        else:
            df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 2 else x)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Data prepared and saved to {output_path}")
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Sentiment distribution: {df['sentiment'].value_counts()}")

    except Exception as e:
        logger.error(f"Error preparing data format: {e}")
        raise e