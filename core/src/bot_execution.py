import os
import sys
import torch
import torch.nn as nn
import json
from transformers import BertTokenizer, BertModel
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import logging
from typing import Dict, Tuple, Any, NoReturn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Update path to the LCF-ATEPC model
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'lcf_atepc')

# Check for symlink to latest model
SYMLINK_PATH = os.path.join(MODELS_DIR, 'lcf_atepc_latest')
if os.path.exists(SYMLINK_PATH) and os.path.islink(SYMLINK_PATH):
    MODEL_PATH = SYMLINK_PATH
    logger.info(f"Using latest model from symlink: {SYMLINK_PATH}")

if not os.path.exists(MODEL_PATH):
    logger.error(f"Model directory not found. Full path: {os.path.abspath(MODEL_PATH)}")
    raise ValueError(f"Model directory not found: {MODEL_PATH}")
else:
    logger.info(f"Model directory found: {os.path.abspath(MODEL_PATH)}")

# LCF-ATEPC model implementation for aspect-based sentiment analysis
class LCF_ATEPC(nn.Module):
    """
    LCF-ATEPC (Local Context Focus for Aspect Term Extraction and Polarity Classification) model.
    
    This model uses BERT as a backbone and adds specialized components for aspect-based sentiment analysis:
    - Self-attention mechanism for aspect-context relationship
    - Local context focus through context window around aspects
    - Feature fusion for combining aspect and context information
    """
    def __init__(self, pretrained_model_name='bert-base-uncased',
                 hidden_size=768, num_aspect_labels=2, num_sentiment_labels=2,
                 context_window=3, dropout_rate=0.15):
        super(LCF_ATEPC, self).__init__()
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
        """
        Forward pass through the LCF-ATEPC model.
        
        Args:
            input_ids: Token ids from tokenizer
            attention_mask: Attention mask from tokenizer
            token_type_ids: Token type ids for segment embeddings
            aspect_positions: List of [start, end] positions for aspects
            
        Returns:
            logits: Sentiment classification logits
        """
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
    """
    Class for analyzing sentiment in text reviews for specific aspects using the LCF-ATEPC model.
    
    This analyzer uses a fine-tuned BERT-based model that has been optimized for
    aspect-based sentiment analysis tasks.
    """
    
    def __init__(self) -> None:
        """
        Initialize the sentiment analyzer with the LCF-ATEPC model and tokenizer.
        
        Loads model configuration and weights from the specified model path.
        Sets up the model on the available device (GPU if available, CPU otherwise).
        
        Raises:
            FileNotFoundError: If model files cannot be found.
            ValueError: If the model configuration is invalid.
            Exception: If there is an error loading the model.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load config
            config_path = os.path.join(MODEL_PATH, "inference_config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
                
            with open(config_path, "r") as f:
                self.config = json.load(f)
            
            logger.info(f"Loading tokenizer from {MODEL_PATH}")
            tokenizer_path = os.path.join(MODEL_PATH, "tokenizer")
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_path}")
                
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            
            logger.info(f"Initializing LCF-ATEPC model")
            self.model = LCF_ATEPC(
                pretrained_model_name=self.config.get("pretrained_model_name", "bert-base-uncased"),
                num_aspect_labels=self.config.get("num_aspect_labels", 2),
                num_sentiment_labels=self.config.get("num_sentiment_labels", 2),
                context_window=self.config.get("context_window", 3),
                dropout_rate=0.1  # Lower dropout for inference
            )
            
            # Load model weights
            model_weights_path = os.path.join(MODEL_PATH, "model.pt")
            if not os.path.exists(model_weights_path):
                raise FileNotFoundError(f"Model weights not found: {model_weights_path}")
                
            logger.info(f"Loading model weights from {model_weights_path}")
            state_dict = torch.load(model_weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # Sentiment mapping from model config
            self.sentiment_map = self.config.get("sentiment_mapping", {"0": "negative", "1": "positive"})
            
            logger.info("Aspect sentiment analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {str(e)}")
            raise
        
    def analyze_sentiment(self, text: str, aspect: str) -> Tuple[str, float]:
        """
        Analyze sentiment for a given aspect in the provided text.
        
        Uses the LCF-ATEPC model to determine sentiment polarity (positive/negative)
        and confidence for the specified aspect within the text.
        
        Args:
            text (str): Review text
            aspect (str): Aspect to analyze
            
        Returns:
            Tuple[str, float]: (sentiment, confidence)
            
        Raises:
            RuntimeError: If an error occurs during inference.
            Exception: If an error occurs during sentiment analysis.
        """
        try:
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
                    # Fallback - estimate position using non-special tokens
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
                
            except Exception as e:
                logger.warning(f"Error finding aspect markers: {e}")
                aspect_indices = [1, 5]  # Default fallback
            
            # Run inference
            with torch.no_grad():
                logits = self.model(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                    token_type_ids=encoding["token_type_ids"],
                    aspect_positions=[[aspect_indices[0], aspect_indices[1]]]
                )
                
                # Apply temperature scaling for calibration
                temperature = 1.2  # Slightly higher temperature for smoother probabilities
                scaled_logits = logits / temperature
                
                probs = torch.softmax(scaled_logits, dim=1)
                prediction = torch.argmax(probs, dim=1).item()
                confidence = probs[0][prediction].item()
                
                sentiment = self.sentiment_map.get(str(prediction), "unknown")
            
            # Apply post-processing rules for better results
            confidence = min(confidence, 0.95)  # Cap confidence to avoid overconfidence
            
            return sentiment, confidence
            
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {str(e)}")
            raise

class TelegramBot:
    """
    Telegram bot for aspect-based sentiment analysis.
    
    This bot allows users to submit reviews along with aspects to analyze
    and returns sentiment analysis results using the AspectSentimentAnalyzer.
    """
    
    def __init__(self, token: str) -> None:
        """
        Initialize the Telegram bot with the provided token.
        
        Args:
            token (str): Telegram Bot API token
            
        Raises:
            ValueError: If token is invalid.
            Exception: If there is an error initializing the analyzer.
        """
        self.token = token
        try:
            self.analyzer = AspectSentimentAnalyzer()
        except Exception as e:
            logger.error(f"Error initializing analyzer: {str(e)}")
            raise
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the /start command.
        
        Sends a welcome message explaining how to use the bot.
        
        Args:
            update (Update): Update object containing message information
            context (ContextTypes.DEFAULT_TYPE): Context object
            
        Raises:
            Exception: If there is an error handling the command.
        """
        welcome_message = (
            "Hello! I'm an aspect-based sentiment analysis bot for reviews.\n\n"
            "Send me a message in this format:\n"
            "review text | aspect\n\n"
            "Example:\n"
            "The movie was great, but the actors performed poorly | actors"
        )
        await update.message.reply_text(welcome_message)
        
    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the /help command.
        
        Provides detailed instructions on how to use the bot.
        
        Args:
            update (Update): Update object containing message information
            context (ContextTypes.DEFAULT_TYPE): Context object
            
        Raises:
            Exception: If there is an error handling the command.
        """
        help_message = (
            "How to use this bot:\n\n"
            "1. Send a message in this format:\n"
            "review text | aspect\n\n"
            "2. I'll analyze the sentiment for the specified aspect\n\n"
            "3. You'll receive a response with the analysis result and confidence level\n\n"
            "Example:\n"
            "The movie was great, but the actors performed poorly | actors\n\n"
            "This bot uses the LCF-ATEPC model, which specializes in aspect-based sentiment analysis."
        )
        await update.message.reply_text(help_message)
        
    async def analyze_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle text messages for sentiment analysis.
        
        Parses the message, extracts review text and aspect,
        performs sentiment analysis, and responds with the result.
        
        Args:
            update (Update): Update object containing message information
            context (ContextTypes.DEFAULT_TYPE): Context object
            
        Raises:
            ValueError: If message format is incorrect.
            Exception: If there is an error processing the message.
        """
        try:
            message_parts = update.message.text.split('|')
            if len(message_parts) != 2:
                await update.message.reply_text(
                    "Please send a message in this format:\n"
                    "review text | aspect"
                )
                return
                
            text, aspect = [part.strip() for part in message_parts]
            
            if not text or not aspect:
                await update.message.reply_text(
                    "Please ensure both the review text and aspect are provided.\n"
                    "Format: review text | aspect"
                )
                return
            
            await update.message.reply_text("Analyzing sentiment... Please wait.")
            
            sentiment, confidence = self.analyzer.analyze_sentiment(text, aspect)
            
            emoji = "👍" if sentiment == "positive" else "👎"
            confidence_text = f"{confidence:.1%}"
            
            response = (
                f"Sentiment analysis for aspect '{aspect}':\n\n"
                f"Result: {sentiment} {emoji}\n"
                f"Confidence: {confidence_text}"
            )
            
            await update.message.reply_text(response)
            
        except ValueError:
            await update.message.reply_text(
                "Error in message format. Please use the separator '|' between text and aspect."
            )
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            await update.message.reply_text(
                "An error occurred during analysis. Please try again."
            )
    
    def run(self) -> NoReturn:
        """
        Run the bot using the Telegram API.
        
        Sets up handlers for commands and messages,
        then starts polling for updates.
        
        Raises:
            Exception: If there is an error running the bot.
        """
        application = Application.builder().token(self.token).build()
        
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.analyze_message))
        
        logger.info("Starting Telegram bot polling...")
        application.run_polling()

if __name__ == "__main__":
    # Replace with your actual bot token from BotFather
    BOT_TOKEN = "7868193869:AAHNN4u15vkq-y2NHrYDBoylgWBS1l5w-4U"
    
    if BOT_TOKEN == "YOUR_BOT_TOKEN":
        logger.warning("You are using a test token. Please replace 'YOUR_BOT_TOKEN' with an actual token from @BotFather.")
    
    try:
        logger.info("Initializing Telegram bot with LCF-ATEPC model")
        bot = TelegramBot(BOT_TOKEN)
        bot.run()
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        sys.exit(1) 