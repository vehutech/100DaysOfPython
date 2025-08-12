# Day 81: Recurrent Neural Networks (RNNs) - AI Mastery Course

## Learning Objective
By the end of this lesson, you will understand how Recurrent Neural Networks process sequential data, implement LSTM and GRU networks using Python, and build sequence-to-sequence models that can handle complex temporal patterns in data.

---

Imagine that you're running a bustling restaurant where orders don't come in isolation, but as part of flowing conversations between customers and waitstaff. Each word spoken builds upon the previous ones, creating meaning through context and sequence. A traditional chef might only focus on individual ingredients, but you need a special kind of culinary memory - one that remembers the flavor combinations from earlier in the meal to create the perfect final dish. This is exactly what Recurrent Neural Networks do with data: they maintain a "memory" of previous information to make sense of sequences.

---

## 1. Vanilla RNNs and Their Limitations

Just like a chef who tries to remember every ingredient used in a complex recipe, vanilla RNNs attempt to carry information forward through time. However, much like how a chef's memory can fade when preparing an elaborate multi-course meal, vanilla RNNs suffer from the "vanishing gradient problem."

### Basic RNN Implementation

```python
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize a basic RNN
        
        Args:
            input_size: Size of input features (like ingredients in a recipe)
            hidden_size: Size of hidden state (chef's working memory)
            output_size: Size of output (final dish categories)
        """
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # The chef's recipe book - transforms input to hidden state
        self.input_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        # The chef's final plating - transforms hidden to output
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        # Activation function - like the chef's taste testing
        self.activation = nn.Tanh()
        
    def forward(self, x, hidden):
        """
        Forward pass through the RNN
        
        Args:
            x: Input tensor (current ingredient)
            hidden: Previous hidden state (chef's memory so far)
        """
        # Combine current input with previous memory
        combined = torch.cat((x, hidden), dim=1)
        # Update the chef's working memory
        hidden = self.activation(self.input_to_hidden(combined))
        # Create the output dish
        output = self.hidden_to_output(hidden)
        return output, hidden
    
    def init_hidden(self, batch_size):
        """Initialize hidden state - like clearing the chef's mind"""
        return torch.zeros(batch_size, self.hidden_size)

# Example usage - predicting next number in sequence
def demonstrate_vanilla_rnn():
    # Create a simple sequence prediction task
    sequence_length = 10
    input_size = 1
    hidden_size = 20
    output_size = 1
    
    # Initialize our RNN chef
    rnn = VanillaRNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = Adam(rnn.parameters(), lr=0.01)
    
    # Generate training data - simple sine wave sequence
    x = np.linspace(0, 4*np.pi, 100)
    y = np.sin(x)
    
    print("Training vanilla RNN on sine wave prediction...")
    print("Each step is like a chef learning to predict the next flavor")
    
    return rnn

# Syntax Explanation:
# - nn.Module: Base class for all neural network modules in PyTorch
# - super().__init__(): Calls parent class constructor
# - nn.Linear(in_features, out_features): Creates a linear transformation layer
# - torch.cat(): Concatenates tensors along specified dimension
# - nn.Tanh(): Hyperbolic tangent activation function
```

### Limitations of Vanilla RNNs

```python
def show_vanishing_gradient_problem():
    """
    Demonstrate why vanilla RNNs struggle with long sequences
    Like a chef trying to remember the first spice used in a 20-course meal
    """
    sequence_lengths = [5, 10, 20, 50]
    gradient_magnitudes = []
    
    for seq_len in sequence_lengths:
        # Simulate gradient flow through time
        # As sequence gets longer, gradients get exponentially smaller
        gradient_magnitude = 0.9 ** seq_len  # Typical gradient decay
        gradient_magnitudes.append(gradient_magnitude)
        
        print(f"Sequence length {seq_len}: Gradient magnitude = {gradient_magnitude:.6f}")
        print(f"  - Like a chef's memory of flavors from {seq_len} steps ago")
    
    # Visualize the problem
    plt.figure(figsize=(10, 6))
    plt.plot(sequence_lengths, gradient_magnitudes, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Sequence Length (Steps Back in Recipe)')
    plt.ylabel('Gradient Magnitude (Memory Strength)')
    plt.title('The Vanishing Gradient Problem')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to show exponential decay
    
    print("\nThe Problem: As sequences get longer, the RNN forgets earlier information")
    print("Solution needed: A better memory system (LSTM/GRU)")
    
    return gradient_magnitudes

show_vanishing_gradient_problem()
```

---

## 2. Long Short-Term Memory (LSTM)

LSTMs are like master chefs with both short-term working memory and a detailed recipe notebook. They can selectively remember important techniques while forgetting irrelevant details, making them perfect for complex culinary sequences.

### LSTM Implementation

```python
class ChefLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        LSTM - Like a master chef with selective memory
        
        Args:
            input_size: Ingredients coming in
            hidden_size: Chef's working memory capacity  
            output_size: Types of dishes to create
            num_layers: Levels of culinary expertise
        """
        super(ChefLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The LSTM layer - master chef's memory system
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Final output layer - plating the dish
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Dropout for regularization - like varying cooking techniques
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, hidden=None):
        """
        Process sequence through LSTM
        
        Args:
            x: Input sequence (batch_size, seq_length, input_size)
            hidden: Previous hidden state (optional)
        """
        # Pass through LSTM layers
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Generate final output
        output = self.output_layer(lstm_out)
        
        return output, hidden

# Demonstrate LSTM components
def explain_lstm_gates():
    """
    Explain LSTM gates using cooking analogies
    """
    print("=== LSTM Gates Explained ===")
    print("\n1. FORGET GATE - The Cleanup Crew")
    print("   - Decides what old cooking techniques to forget")
    print("   - Like clearing expired ingredients from memory")
    print("   - Formula: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)")
    
    print("\n2. INPUT GATE - The Ingredient Selector")  
    print("   - Chooses which new information to store")
    print("   - Like deciding which new spices are worth remembering")
    print("   - Formula: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)")
    
    print("\n3. CANDIDATE VALUES - The Recipe Creator")
    print("   - Creates new potential memories")
    print("   - Like experimenting with new flavor combinations")
    print("   - Formula: C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)")
    
    print("\n4. CELL STATE UPDATE - The Recipe Book")
    print("   - Updates long-term memory")
    print("   - Formula: C_t = f_t * C_{t-1} + i_t * C̃_t")
    
    print("\n5. OUTPUT GATE - The Plating Decision")
    print("   - Decides what to output from current memory")
    print("   - Like choosing which flavors to highlight in the final dish")
    print("   - Formula: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)")

explain_lstm_gates()

# Practical LSTM example
def create_sequence_predictor():
    """
    Create an LSTM that predicts cooking temperatures
    """
    # Generate synthetic cooking temperature data
    def generate_cooking_sequence(length=100):
        # Simulate cooking temperature over time
        time = np.linspace(0, 10, length)
        base_temp = 180 + 30 * np.sin(0.5 * time)  # Base oven temperature
        noise = np.random.normal(0, 5, length)      # Random fluctuations
        return base_temp + noise
    
    # Create training data
    temperatures = generate_cooking_sequence(200)
    
    # Prepare sequences for training
    def create_sequences(data, seq_length=10):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(temperatures)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X).unsqueeze(-1)  # Add feature dimension
    y_tensor = torch.FloatTensor(y).unsqueeze(-1)
    
    print(f"Training data shape: {X_tensor.shape}")
    print(f"Target data shape: {y_tensor.shape}")
    print("Each sequence represents 10 temperature readings")
    print("Goal: Predict the next temperature reading")
    
    return X_tensor, y_tensor

create_sequence_predictor()
```

---

## 3. Gated Recurrent Units (GRU)

GRUs are like efficient sous chefs - they achieve similar results to LSTMs but with a simpler approach. They have fewer "utensils" (parameters) but can still handle complex recipes effectively.

### GRU Implementation

```python
class SousChefGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        """
        GRU - The efficient sous chef
        Simpler than LSTM but still very capable
        """
        super(SousChefGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layers - efficient memory management
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=0.2)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        # Process through GRU layers
        out, hidden = self.gru(x, hidden)
        
        # Take only the last output for prediction
        out = self.fc(out[:, -1, :])  # Use last time step
        
        return out, hidden

def compare_gru_lstm():
    """
    Compare GRU and LSTM architectures
    """
    print("=== GRU vs LSTM Comparison ===")
    print("\nGRU (Sous Chef):")
    print("✓ 2 gates: Reset gate, Update gate")
    print("✓ Fewer parameters (faster training)")
    print("✓ Simpler architecture")
    print("✓ Good for smaller datasets")
    
    print("\nLSTM (Master Chef):")
    print("✓ 3 gates: Forget, Input, Output")
    print("✓ More parameters (more expressive)")
    print("✓ Separate cell state and hidden state")
    print("✓ Better for complex, long sequences")
    
    # Parameter count comparison
    input_size, hidden_size = 50, 128
    
    # Create models
    gru_model = SousChefGRU(input_size, hidden_size, 1)
    lstm_model = ChefLSTM(input_size, hidden_size, 1)
    
    # Count parameters
    gru_params = sum(p.numel() for p in gru_model.parameters())
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    
    print(f"\nParameter Count:")
    print(f"GRU: {gru_params:,} parameters")
    print(f"LSTM: {lstm_params:,} parameters")
    print(f"LSTM has {lstm_params/gru_params:.1f}x more parameters")

compare_gru_lstm()

# Syntax Explanation:
# - nn.GRU(): PyTorch's GRU implementation
# - batch_first=True: Input shape is (batch, seq, features)
# - dropout=0.2: Applies 20% dropout between GRU layers
# - out[:, -1, :]: Selects the last time step from each sequence
# - p.numel(): Returns number of elements in parameter tensor
```

---

## 4. Sequence-to-Sequence Models

Sequence-to-sequence models are like having a translator chef who understands one cuisine and can recreate its essence in a completely different culinary tradition. They encode the "essence" of an input sequence and decode it into a different output sequence.

### Seq2Seq Implementation

```python
class EncoderChef(nn.Module):
    """
    Encoder - Understands the input recipe sequence
    Like a chef analyzing a foreign dish to understand its essence
    """
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(EncoderChef, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.3)
        
    def forward(self, x):
        # Process the entire input sequence
        outputs, (hidden, cell) = self.lstm(x)
        # Return final hidden state as the "recipe essence"
        return hidden, cell

class DecoderChef(nn.Module):
    """
    Decoder - Creates output sequence from encoded essence  
    Like a chef recreating a dish in their own style
    """
    def __init__(self, hidden_size, output_size, num_layers=2):
        super(DecoderChef, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.3)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, x, hidden, cell):
        # Generate one step of output sequence
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        output = self.softmax(self.out(output))
        return output, hidden, cell

class Seq2SeqChef(nn.Module):
    """
    Complete Sequence-to-Sequence model
    Like a master chef who can translate between culinary languages
    """
    def __init__(self, encoder_input_size, decoder_output_size, hidden_size):
        super(Seq2SeqChef, self).__init__()
        self.encoder = EncoderChef(encoder_input_size, hidden_size)
        self.decoder = DecoderChef(hidden_size, decoder_output_size)
        
    def forward(self, source_sequence, target_sequence=None, max_length=20):
        batch_size = source_sequence.size(0)
        
        # Encode the source sequence
        encoder_hidden, encoder_cell = self.encoder(source_sequence)
        
        # Initialize decoder
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        
        # Start with a special "start of sequence" token
        decoder_input = torch.zeros(batch_size, 1, self.decoder.output_size)
        
        outputs = []
        
        # Generate output sequence step by step
        for i in range(max_length):
            # Generate next output
            output, decoder_hidden, decoder_cell = self.decoder(
                decoder_input, decoder_hidden, decoder_cell
            )
            outputs.append(output)
            
            # Use current output as next input (teacher forcing in training)
            if target_sequence is not None and i < target_sequence.size(1) - 1:
                decoder_input = target_sequence[:, i:i+1]  # Teacher forcing
            else:
                decoder_input = output  # Use model's own output
                
        return torch.cat(outputs, dim=1)

# Demonstrate attention mechanism concept
def explain_attention_mechanism():
    """
    Explain attention - like a chef's selective focus
    """
    print("=== Attention Mechanism ===")
    print("\nImagine a chef preparing a fusion dish:")
    print("1. They taste the original dish (encoder)")
    print("2. While cooking the fusion version, they selectively")
    print("   pay attention to different aspects of the original")
    print("3. Some flavors need more attention than others")
    print("\nThis is exactly what attention does:")
    print("- Allows decoder to 'look back' at all encoder states")
    print("- Learns which parts of input are most relevant")
    print("- Solves the bottleneck problem of vanilla seq2seq")
    
    # Simple attention weight visualization
    sequence_length = 10
    attention_weights = torch.softmax(torch.randn(1, sequence_length), dim=1)
    
    print(f"\nExample attention weights across input sequence:")
    for i, weight in enumerate(attention_weights[0]):
        bar = "█" * int(weight * 20)  # Visual bar
        print(f"Position {i+1}: {weight:.3f} {bar}")
    
    print("\nHigher weights = more attention to that input position")

explain_attention_mechanism()

# Syntax Explanation:
# - LogSoftmax(dim=-1): Applies log-softmax along last dimension
# - torch.cat(tensors, dim=1): Concatenates along dimension 1
# - size(0): Returns batch size (first dimension)
# - teacher_forcing: Using ground truth as input during training
```

---

## Final Project: Recipe Style Transfer System

Now let's combine everything into a practical project that demonstrates all RNN concepts.

### Complete Recipe Style Transfer Implementation

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

class RecipeDataset(Dataset):
    """
    Dataset for recipe style transfer
    Input: Simple recipe, Output: Gourmet version
    """
    def __init__(self, simple_recipes, gourmet_recipes, vocab_size=1000):
        self.simple_recipes = simple_recipes
        self.gourmet_recipes = gourmet_recipes
        
        # Build vocabulary from both recipe types
        all_text = simple_recipes + gourmet_recipes
        self.vocab = self.build_vocabulary(all_text, vocab_size)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
    def build_vocabulary(self, texts, vocab_size):
        """Build vocabulary from recipe texts"""
        words = []
        for text in texts:
            # Simple tokenization
            words.extend(re.findall(r'\w+', text.lower()))
        
        # Get most common words
        word_counts = Counter(words)
        most_common = word_counts.most_common(vocab_size - 2)  # Reserve space for special tokens
        
        vocab = ['<PAD>', '<UNK>'] + [word for word, _ in most_common]
        return vocab
    
    def text_to_sequence(self, text, max_length=50):
        """Convert text to sequence of indices"""
        words = re.findall(r'\w+', text.lower())
        sequence = [self.word_to_idx.get(word, 1) for word in words]  # 1 is <UNK>
        
        # Pad or truncate to max_length
        if len(sequence) < max_length:
            sequence += [0] * (max_length - len(sequence))  # 0 is <PAD>
        else:
            sequence = sequence[:max_length]
            
        return sequence
    
    def __len__(self):
        return len(self.simple_recipes)
    
    def __getitem__(self, idx):
        simple_seq = self.text_to_sequence(self.simple_recipes[idx])
        gourmet_seq = self.text_to_sequence(self.gourmet_recipes[idx])
        
        return torch.LongTensor(simple_seq), torch.LongTensor(gourmet_seq)

class RecipeStyleTransfer(nn.Module):
    """
    Complete recipe style transfer model
    Converts simple recipes to gourmet versions
    """
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=512):
        super(RecipeStyleTransfer, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        # Shared embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Encoder (analyzes simple recipe)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_size, 
                                   num_layers=2, batch_first=True, dropout=0.3)
        
        # Decoder (generates gourmet recipe)  
        self.decoder_lstm = nn.LSTM(embedding_dim, hidden_size,
                                   num_layers=2, batch_first=True, dropout=0.3)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, 
                                              batch_first=True, dropout=0.1)
        
    def forward(self, simple_recipe, gourmet_recipe=None, max_length=50):
        batch_size = simple_recipe.size(0)
        
        # Encode simple recipe
        simple_embedded = self.embedding(simple_recipe)
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder_lstm(simple_embedded)
        
        # Initialize decoder
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        
        # Generate gourmet recipe
        if gourmet_recipe is not None:
            # Training mode - use teacher forcing
            gourmet_embedded = self.embedding(gourmet_recipe)
            decoder_outputs, _ = self.decoder_lstm(gourmet_embedded, 
                                                  (decoder_hidden, decoder_cell))
            
            # Apply attention
            attended_output, attention_weights = self.attention(
                decoder_outputs, encoder_outputs, encoder_outputs
            )
            
            # Generate final output
            output = self.output_projection(attended_output)
            return output, attention_weights
        
        else:
            # Inference mode - generate step by step
            outputs = []
            current_input = torch.zeros(batch_size, 1, dtype=torch.long)
            
            for _ in range(max_length):
                # Embed current input
                embedded = self.embedding(current_input)
                
                # Pass through decoder
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder_lstm(
                    embedded, (decoder_hidden, decoder_cell)
                )
                
                # Apply attention
                attended_output, _ = self.attention(
                    decoder_output, encoder_outputs, encoder_outputs
                )
                
                # Generate next token
                output = self.output_projection(attended_output)
                outputs.append(output)
                
                # Use output as next input
                current_input = torch.argmax(output, dim=-1)
            
            return torch.cat(outputs, dim=1)

# Training function
def train_recipe_transfer_model():
    """
    Train the recipe style transfer model
    """
    # Sample data (in practice, you'd load from files)
    simple_recipes = [
        "boil pasta add tomato sauce and cheese",
        "fry chicken with salt and pepper", 
        "mix salad with dressing",
        "bake potato with butter",
        "scramble eggs with milk"
    ]
    
    gourmet_recipes = [
        "prepare al dente pasta with artisanal tomato reduction and aged parmesan",
        "pan sear free range chicken breast with herb crusted seasoning",
        "toss organic mixed greens with champagne vinaigrette", 
        "roast fingerling potatoes with truffle butter and rosemary",
        "whisk farm fresh eggs with heavy cream and chive garnish"
    ]
    
    # Create dataset
    dataset = RecipeDataset(simple_recipes, gourmet_recipes)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize model
    model = RecipeStyleTransfer(len(dataset.vocab))
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("=== Training Recipe Style Transfer Model ===")
    print(f"Vocabulary size: {len(dataset.vocab)}")
    print(f"Training samples: {len(dataset)}")
    
    # Training loop
    model.train()
    for epoch in range(10):  # Limited epochs for demo
        total_loss = 0
        for batch_idx, (simple, gourmet) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            output, attention_weights = model(simple, gourmet[:, :-1])  # Exclude last token
            
            # Calculate loss
            target = gourmet[:, 1:].contiguous()  # Exclude first token, shift left
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/10, Average Loss: {avg_loss:.4f}")
    
    print("\nModel training complete!")
    return model, dataset

# Demonstration
model, dataset = train_recipe_transfer_model()

# Syntax Explanation:
# - nn.Embedding(num_embeddings, embedding_dim, padding_idx=0): Creates embedding layer
# - nn.MultiheadAttention(): PyTorch's multi-head attention implementation  
# - ignore_index=0: CrossEntropyLoss ignores padding tokens (index 0)
# - contiguous(): Ensures tensor memory is contiguous for view operations
# - view(-1, size): Reshapes tensor, -1 infers dimension size
# - torch.argmax(tensor, dim=-1): Returns indices of maximum values
```

---

## Assignment: Sentiment Analysis with Bidirectional LSTM

Create a bidirectional LSTM model that analyzes restaurant reviews and predicts sentiment scores. Your model should process review text from both directions (like reading a recipe forwards and backwards to fully understand it) and output a sentiment score between 0 (negative) and 1 (positive).

### Requirements:

1. **Data Preparation**: Create a dataset of restaurant reviews with sentiment labels
2. **Model Architecture**: Implement a bidirectional LSTM with the following components:
   - Embedding layer for text processing
   - Bidirectional LSTM layers (2 layers minimum)
   - Attention mechanism to focus on important words
   - Output layer for sentiment prediction

3. **Training Process**: 
   - Use appropriate loss function and optimizer
   - Implement training loop with validation
   - Track and visualize training progress

4. **Evaluation**: Test your model on sample reviews and analyze which words the attention mechanism focuses on

### Starter Code Structure:
```python
class BidirectionalSentimentAnalyzer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        # Your implementation here
        pass
    
    def forward(self, x):
        # Your implementation here  
        pass

# Create sample restaurant reviews dataset
reviews = [
    "The food was absolutely delicious and the service was outstanding",
    "Terrible experience, cold food and rude staff",
    # Add more examples...
]

labels = [1, 0, ...]  # 1 for positive, 0 for negative

# Train and evaluate your model
```

### Expected Deliverables:
- Complete working model implementation
- Training script with progress visualization  
- Analysis of attention weights on test examples
- Brief report (200 words) explaining your architecture choices

---

## Summary

In this lesson, you've learned how RNNs process sequential data like skilled chefs handling complex recipes. You've implemented vanilla RNNs (with their memory limitations), powerful LSTMs (master chefs with selective memory), efficient GRUs (capable sous chefs), and sequence-to-sequence models (culinary translators). The final project demonstrated how these concepts combine to create practical applications for text processing and generation.

**Key Takeaways:**
- RNNs maintain memory across time steps but suffer from vanishing gradients
- LSTMs use gates to selectively remember and forget information
- GRUs provide similar capabilities to LSTMs with fewer parameters
- Seq2seq models can transform one sequence type into another
- Attention mechanisms allow models to focus on relevant input parts

The assignment will help you practice these concepts by building a bidirectional sentiment analyzer, reinforcing your understanding of how RNNs process sequential data in both directions.