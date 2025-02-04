# Attention-Enhanced Image Captioning System

An end-to-end neural network system that generates natural language descriptions for images by combining computer vision and natural language processing techniques with an attention mechanism.

## Project Highlights
* Implemented a sophisticated attention mechanism with Bidirectional LSTM that enables the model to focus on relevant image features while generating each word of the caption, enhancing the quality and relevance of generated descriptions.
* Leveraged transfer learning through a pre-trained CNN for image feature extraction, combined with an efficient text vectorization pipeline that handles multiple captions (5 per image) for robust training.
* Designed an optimized training pipeline using teacher forcing and proper sequence masking, allowing the model to effectively learn the temporal dependencies in caption generation while handling variable-length sequences.
* Built with a robust architecture incorporating BatchNormalization and Dropout layers for regularization, ensuring model stability and preventing overfitting during the training process.

## Model Performance
BLEU Scores on validation set:
- BLEU-1: 0.4908 (unigram matching)
- BLEU-2: 0.2550 (bigram matching)
- BLEU-3: 0.1232 (trigram matching)
- BLEU-4: 0.0634 (4-gram matching)

## Dataset
This project uses the Flickr8k dataset, which contains:
- 8,000 images
- 5 captions per image
- Total of 40,000 captions

## Model Architecture
1. Image Processing:
   - Pre-trained CNN for feature extraction
   - Global Max Pooling
   - Dense layers with ReLU activation

2. Caption Processing:
   - Embedding layer with masking
   - Bidirectional LSTM
   - Attention mechanism
   - Dropout and BatchNormalization for regularization

3. Attention Mechanism:
   - Dense layers with tanh activation
   - Softmax for attention weights
   - Weighted sum for context vector

## Requirements
- TensorFlow 2.x
- NumPy
- NLTK (for BLEU score evaluation)
- PIL (for image processing)

## Usage
1. Prepare the dataset:
   - Download Flickr8k dataset
   - Place images in the Images/ directory
   - Place captions in captions.txt

2. Run the training:
   ```python
   python train.ipynb
