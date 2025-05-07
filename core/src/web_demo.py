import os
import sys
import json
import logging
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project configuration
from config.config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Model instances
model_lcf_atepc = None
model_roberta = None

def load_models():
    """
    Load models for inference.
    
    This function loads both LCF-ATEPC and RoBERTa models
    for aspect-based sentiment analysis from their respective
    saved paths.
    """
    global model_lcf_atepc, model_roberta
    
    # Load LCF-ATEPC model
    try:
        from models.lcf_atepc.inference import AspectSentimentAnalyzer
        lcf_atepc_path = os.path.join(CONFIG["lcf_atepc_output_path"], "best_model")
        if os.path.exists(lcf_atepc_path):
            model_lcf_atepc = AspectSentimentAnalyzer(lcf_atepc_path)
            logger.info("LCF-ATEPC model successfully loaded")
    except Exception as e:
        logger.error(f"Error loading LCF-ATEPC model: {e}")
    
    # Load RoBERTa model
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        
        roberta_path = CONFIG["roberta_final_path"]
        if os.path.exists(roberta_path):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            tokenizer = AutoTokenizer.from_pretrained(roberta_path)
            model = AutoModelForSequenceClassification.from_pretrained(roberta_path).to(device)
            model.eval()
            
            # Create closure for convenient usage
            def analyze(text, aspect):
                inputs = tokenizer(f"{text} [SEP] {aspect}", 
                                  padding='max_length', 
                                  truncation=True, 
                                  max_length=128, 
                                  return_tensors='pt').to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    prediction = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][prediction].item()
                
                id2label = {0: 'negative', 1: 'positive'}
                return {
                    "sentiment": id2label[prediction],
                    "confidence": float(confidence),
                    "probabilities": {
                        "negative": float(probs[0][0]), 
                        "positive": float(probs[0][1])
                    }
                }
            
            model_roberta = analyze
            logger.info("RoBERTa model successfully loaded")
    except Exception as e:
        logger.error(f"Error loading RoBERTa model: {e}")

@app.route('/')
def index():
    """Render the main page of the demo."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze text and aspect using the loaded models.
    
    This endpoint receives a JSON with text and aspect fields,
    runs the analysis through both models, and returns the predictions.
    """
    try:
        # Get data from request
        data = request.get_json()
        text = data.get('text', '')
        aspect = data.get('aspect', '')
        
        if not text or not aspect:
            return jsonify({'error': 'Text and aspect must be provided'}), 400
        
        # Model results
        results = {}
        
        # Analyze with LCF-ATEPC
        if model_lcf_atepc:
            try:
                results['lcf_atepc'] = model_lcf_atepc.analyze(text, aspect)
            except Exception as e:
                results['lcf_atepc'] = {'error': str(e)}
        
        # Analyze with RoBERTa
        if model_roberta:
            try:
                results['roberta'] = model_roberta(text, aspect)
            except Exception as e:
                results['roberta'] = {'error': str(e)}
                
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/examples')
def examples():
    """
    Return a list of examples for demonstration.
    
    Provides pre-defined movie review examples with various aspects
    that demonstrate the capabilities of the ABSA models.
    """
    examples = [
        {
            "text": "The film's story was boring and predictable, but the visuals were absolutely stunning.",
            "aspects": ["story", "visuals", "acting", "screenplay"]
        },
        {
            "text": "While the acting was superb, the plot had too many holes and inconsistencies to be believable.",
            "aspects": ["acting", "plot", "direction", "characters"]
        },
        {
            "text": "The dialogue was witty and engaging throughout, despite the film's otherwise slow pacing.",
            "aspects": ["dialogue", "pacing", "screenplay", "humor"]
        },
        {
            "text": "The film's soundtrack perfectly complemented the emotional moments, but the special effects were dated and unconvincing.",
            "aspects": ["soundtrack", "special effects", "cinematography", "editing"]
        },
        {
            "text": "The director's vision was bold and innovative, though some scenes dragged on for too long.",
            "aspects": ["director", "scenes", "pacing", "editing"]
        }
    ]
    
    return jsonify(examples)

@app.route('/templates/index.html')
def get_template():
    """Return the HTML template for the demonstration."""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>IMDB ABSA Demo</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .sentiment-positive {
                color: green;
                font-weight: bold;
            }
            .sentiment-negative {
                color: red;
                font-weight: bold;
            }
            .progress-bar-wrapper {
                width: 100%;
                background-color: #f0f0f0;
                border-radius: 5px;
                margin: 5px 0;
            }
            .progress-bar {
                height: 20px;
                border-radius: 5px;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            .progress-bar-positive {
                background-color: #28a745;
            }
            .progress-bar-negative {
                background-color: #dc3545;
            }
            .model-card {
                margin-bottom: 20px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body>
        <div class="container my-5">
            <h1 class="text-center mb-4">IMDB Aspect-Based Sentiment Analysis Demo</h1>
            
            <div class="row mb-4">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h2 class="h5 mb-0">Review Analysis</h2>
                        </div>
                        <div class="card-body">
                            <div class="mb-3">
                                <label for="review-text" class="form-label">Movie Review Text:</label>
                                <textarea id="review-text" class="form-control" rows="4" placeholder="Enter movie review here..."></textarea>
                            </div>
                            <div class="mb-3">
                                <label for="aspect" class="form-label">Aspect to Analyze:</label>
                                <input type="text" class="form-control" id="aspect" placeholder="e.g. acting, plot, visuals">
                            </div>
                            <button id="analyze-btn" class="btn btn-primary">Analyze</button>
                            <button id="load-example-btn" class="btn btn-secondary">Load Example</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div id="results-section" class="row mb-4" style="display: none;">
                <div class="col-md-12">
                    <h3 class="mb-3">Analysis Results</h3>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card model-card">
                                <div class="card-header">
                                    <h4 class="h5 mb-0">LCF-ATEPC Model</h4>
                                </div>
                                <div class="card-body">
                                    <div id="lcf-atepc-results">
                                        <p>Sentiment: <span id="lcf-atepc-sentiment"></span></p>
                                        <p>Confidence: <span id="lcf-atepc-confidence"></span>%</p>
                                        <div class="progress-bar-wrapper">
                                            <div id="lcf-atepc-pos-bar" class="progress-bar progress-bar-positive"></div>
                                        </div>
                                        <div class="progress-bar-wrapper">
                                            <div id="lcf-atepc-neg-bar" class="progress-bar progress-bar-negative"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card model-card">
                                <div class="card-header">
                                    <h4 class="h5 mb-0">RoBERTa Model</h4>
                                </div>
                                <div class="card-body">
                                    <div id="roberta-results">
                                        <p>Sentiment: <span id="roberta-sentiment"></span></p>
                                        <p>Confidence: <span id="roberta-confidence"></span>%</p>
                                        <div class="progress-bar-wrapper">
                                            <div id="roberta-pos-bar" class="progress-bar progress-bar-positive"></div>
                                        </div>
                                        <div class="progress-bar-wrapper">
                                            <div id="roberta-neg-bar" class="progress-bar progress-bar-negative"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div id="examples-accordion" class="accordion mb-5"></div>
            
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h2 class="h5 mb-0">About This Demo</h2>
                        </div>
                        <div class="card-body">
                            <p>This demo showcases Aspect-Based Sentiment Analysis (ABSA) for movie reviews. Unlike traditional sentiment analysis, ABSA determines sentiment specifically toward particular aspects of a movie (like "acting", "visuals", "plot", etc.).</p>
                            <p>The models compare:</p>
                            <ul>
                                <li><strong>LCF-ATEPC</strong>: A BERT-based model with local context focus that is optimized for aspect sentiment analysis.</li>
                                <li><strong>RoBERTa</strong>: A RoBERTa-based model fine-tuned on movie reviews for aspect sentiment classification.</li>
                            </ul>
                            <p>Try entering a movie review and specify an aspect to analyze its sentiment!</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                // Load examples for the accordion
                fetch('/examples')
                    .then(response => response.json())
                    .then(examples => {
                        const accordionContainer = document.getElementById('examples-accordion');
                        
                        examples.forEach((example, index) => {
                            const accordionItem = document.createElement('div');
                            accordionItem.className = 'accordion-item';
                            
                            const headerId = `heading-${index}`;
                            const collapseId = `collapse-${index}`;
                            
                            accordionItem.innerHTML = `
                                <h2 class="accordion-header" id="${headerId}">
                                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#${collapseId}" aria-expanded="false" aria-controls="${collapseId}">
                                        Example ${index + 1}: ${example.text.substring(0, 70)}...
                                    </button>
                                </h2>
                                <div id="${collapseId}" class="accordion-collapse collapse" aria-labelledby="${headerId}" data-bs-parent="#examples-accordion">
                                    <div class="accordion-body">
                                        <p>${example.text}</p>
                                        <p><strong>Aspects:</strong> ${example.aspects.join(', ')}</p>
                                        <button class="btn btn-sm btn-primary use-example-btn" data-text="${example.text}" data-aspect="${example.aspects[0]}">Use This Example</button>
                                    </div>
                                </div>
                            `;
                            
                            accordionContainer.appendChild(accordionItem);
                        });
                        
                        // Add event listeners to the "Use Example" buttons
                        document.querySelectorAll('.use-example-btn').forEach(button => {
                            button.addEventListener('click', function() {
                                const text = this.getAttribute('data-text');
                                const aspect = this.getAttribute('data-aspect');
                                
                                document.getElementById('review-text').value = text;
                                document.getElementById('aspect').value = aspect;
                                
                                // Scroll to the top of the form
                                document.getElementById('review-text').scrollIntoView({ behavior: 'smooth' });
                            });
                        });
                    })
                    .catch(error => console.error('Error loading examples:', error));
                
                // Analyze button click handler
                document.getElementById('analyze-btn').addEventListener('click', function() {
                    const text = document.getElementById('review-text').value.trim();
                    const aspect = document.getElementById('aspect').value.trim();
                    
                    if (!text || !aspect) {
                        alert('Please enter both review text and aspect to analyze.');
                        return;
                    }
                    
                    analyzeReview(text, aspect);
                });
                
                // Load example button click handler
                document.getElementById('load-example-btn').addEventListener('click', function() {
                    fetch('/examples')
                        .then(response => response.json())
                        .then(examples => {
                            const randomExample = examples[Math.floor(Math.random() * examples.length)];
                            const randomAspect = randomExample.aspects[Math.floor(Math.random() * randomExample.aspects.length)];
                            
                            document.getElementById('review-text').value = randomExample.text;
                            document.getElementById('aspect').value = randomAspect;
                        })
                        .catch(error => console.error('Error loading random example:', error));
                });
                
                // Function to analyze review
                function analyzeReview(text, aspect) {
                    // Show loading state
                    document.getElementById('analyze-btn').disabled = true;
                    document.getElementById('analyze-btn').textContent = 'Analyzing...';
                    
                    fetch('/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ text, aspect })
                    })
                    .then(response => response.json())
                    .then(data => {
                        // Show results section
                        document.getElementById('results-section').style.display = 'block';
                        
                        // LCF-ATEPC results
                        if (data.lcf_atepc && !data.lcf_atepc.error) {
                            const lcfAtepc = data.lcf_atepc;
                            const sentiment = lcfAtepc.sentiment;
                            const confidence = Math.round(lcfAtepc.confidence * 100);
                            
                            const sentimentElement = document.getElementById('lcf-atepc-sentiment');
                            sentimentElement.textContent = sentiment;
                            sentimentElement.className = `sentiment-${sentiment.toLowerCase()}`;
                            
                            document.getElementById('lcf-atepc-confidence').textContent = confidence;
                            
                            // Probabilities
                            const posProb = lcfAtepc.probabilities?.positive || 0;
                            const negProb = lcfAtepc.probabilities?.negative || 0;
                            
                            document.getElementById('lcf-atepc-pos-bar').style.width = `${posProb * 100}%`;
                            document.getElementById('lcf-atepc-pos-bar').textContent = `Positive: ${Math.round(posProb * 100)}%`;
                            
                            document.getElementById('lcf-atepc-neg-bar').style.width = `${negProb * 100}%`;
                            document.getElementById('lcf-atepc-neg-bar').textContent = `Negative: ${Math.round(negProb * 100)}%`;
                        } else {
                            document.getElementById('lcf-atepc-results').innerHTML = '<p class="text-danger">Error analyzing with LCF-ATEPC model</p>';
                        }
                        
                        // RoBERTa results
                        if (data.roberta && !data.roberta.error) {
                            const roberta = data.roberta;
                            const sentiment = roberta.sentiment;
                            const confidence = Math.round(roberta.confidence * 100);
                            
                            const sentimentElement = document.getElementById('roberta-sentiment');
                            sentimentElement.textContent = sentiment;
                            sentimentElement.className = `sentiment-${sentiment.toLowerCase()}`;
                            
                            document.getElementById('roberta-confidence').textContent = confidence;
                            
                            // Probabilities
                            const posProb = roberta.probabilities?.positive || 0;
                            const negProb = roberta.probabilities?.negative || 0;
                            
                            document.getElementById('roberta-pos-bar').style.width = `${posProb * 100}%`;
                            document.getElementById('roberta-pos-bar').textContent = `Positive: ${Math.round(posProb * 100)}%`;
                            
                            document.getElementById('roberta-neg-bar').style.width = `${negProb * 100}%`;
                            document.getElementById('roberta-neg-bar').textContent = `Negative: ${Math.round(negProb * 100)}%`;
                        } else {
                            document.getElementById('roberta-results').innerHTML = '<p class="text-danger">Error analyzing with RoBERTa model</p>';
                        }
                        
                        // Scroll to results
                        document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
                    })
                    .catch(error => {
                        console.error('Error analyzing review:', error);
                        alert('Error analyzing review. Please try again.');
                    })
                    .finally(() => {
                        // Reset button state
                        document.getElementById('analyze-btn').disabled = false;
                        document.getElementById('analyze-btn').textContent = 'Analyze';
                    });
                }
            });
        </script>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    # Load models before starting the server
    load_models()
    
    # Define templates path
    app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    os.makedirs(app.template_folder, exist_ok=True)
    
    # Write HTML template to file
    with open(os.path.join(app.template_folder, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(get_template())
    
    # Start web server
    logger.info("Starting IMDB ABSA Demo server...")
    app.run(debug=True, host='0.0.0.0', port=5000) 