import os

# Prompt 1 to prepend to each file (excludes number 5)
prompt_1 = """AI Mastery Course - 
day ? 
topic ?
1, 2, 3, 4, 5
from this outline, create an actual course:
start with imagine that.....
use the kitchen and chef analogy in the lessons
start with a learning objective.
ensure to include code examples, with a final quality project that locks in concepts learnt.
plus explain the coding syntax you used
add one good assignment. just one.
kindly prioritize or rank as important the following instruction:
1. Do only 1-4, exclude number 5
2. Make sure that the assignment you would give is different from project at number 5
"""

# Prompt 2 to prepend to each file (only number 5)
prompt_2 = """AI Mastery Course - 
day ? 
topic ?
1, 2, 3, 4, 5
from this outline, create an actual course:
start with imagine that.....
use the kitchen and chef analogy in the lessons
start with a learning objective.
ensure to include code examples, with a final quality project that locks in concepts learnt.
plus explain the coding syntax you used
add one good assignment. just one.
kindly prioritize strictly or rank as important the following prompts | instruction:
1. Do only number 5.
2. Exclude number 1-4, i already have it.
3. Exclude non-project related writeups like: syntax explanations, introduction, learning objective, imagine that.. i already have it
4. Ensure to strictly exclude giving an assignment
"""

# Dictionary containing the content for AI Mastery Course (Days 70-100)
day_content = {
    70: """### Day 70: AI & ML Fundamentals
1. Introduction to Artificial Intelligence concepts
2. Machine Learning vs Deep Learning vs AI
3. Types of ML: Supervised, Unsupervised, Reinforcement
4. Python for AI: NumPy, Pandas, Matplotlib
5. **Build**: Data analysis pipeline for ML""",
    
    71: """### Day 71: Statistics & Probability for AI
1. Descriptive and inferential statistics
2. Probability distributions
3. Bayes' theorem and applications
4. Hypothesis testing
5. **Build**: Statistical analysis tool""",
    
    72: """### Day 72: Data Preprocessing & Feature Engineering
1. Data cleaning and preprocessing
2. Handling missing data and outliers
3. Feature scaling and normalization
4. Feature selection and extraction
5. **Build**: Complete data preprocessing pipeline""",
    
    73: """### Day 73: Supervised Learning Algorithms
1. Linear and logistic regression
2. Decision trees and random forests
3. Support Vector Machines (SVM)
4. k-Nearest Neighbors (k-NN)
5. **Build**: Multi-algorithm classifier comparison""",
    
    74: """### Day 74: Model Evaluation & Validation
1. Train/validation/test splits
2. Cross-validation techniques
3. Performance metrics (accuracy, precision, recall, F1)
4. ROC curves and AUC
5. **Build**: Model evaluation framework""",
    
    75: """### Day 75: Unsupervised Learning
1. K-means clustering
2. Hierarchical clustering
3. Principal Component Analysis (PCA)
4. DBSCAN and anomaly detection
5. **Build**: Customer segmentation system""",
    
    76: """### Day 76: Ensemble Methods
1. Bagging and boosting
2. Random Forest deep dive
3. Gradient Boosting (XGBoost, LightGBM)
4. Voting and stacking classifiers
5. **Build**: Ensemble model for competition""",
    
    77: """### Day 77: Scikit-learn Mastery
1. Advanced scikit-learn techniques
2. Pipeline creation and automation
3. Hyperparameter tuning with GridSearch
4. Custom transformers and estimators
5. **Build**: End-to-end ML pipeline""",
    
    78: """### Day 78: Neural Networks Foundations
1. Perceptrons and multi-layer networks
2. Activation functions and backpropagation
3. Gradient descent optimization
4. Setting up TensorFlow/Keras
5. **Build**: Neural network from scratch""",
    
    79: """### Day 79: Deep Neural Networks
1. Deep network architectures
2. Vanishing gradient problem
3. Batch normalization and dropout
4. Advanced optimizers (Adam, RMSprop)
5. **Build**: Deep classifier for image recognition""",
    
    80: """### Day 80: Convolutional Neural Networks (CNNs)
1. Convolution and pooling operations
2. CNN architectures (LeNet, AlexNet, VGG)
3. Transfer learning with pre-trained models
4. Data augmentation techniques
5. **Build**: Complete image recognition app with web interface""",
    
    81: """### Day 81: Recurrent Neural Networks (RNNs)
1. Vanilla RNNs and their limitations
2. Long Short-Term Memory (LSTM)
3. Gated Recurrent Units (GRU)
4. Sequence-to-sequence models
5. **Build**: Text sentiment analyzer""",
    
    82: """### Day 82: Advanced CNN Architectures
1. ResNet and skip connections
2. Inception networks
3. MobileNet and EfficientNet
4. Object detection basics (YOLO, R-CNN)
5. **Build**: Advanced image classifier with multiple architectures""",
    
    83: """### Day 83: Natural Language Processing with Deep Learning
1. Word embeddings (Word2Vec, GloVe)
2. Text preprocessing for NLP
3. RNNs for text classification
4. Attention mechanisms introduction
5. **Build**: Text classification system""",
    
    84: """### Day 84: Generative Models Introduction
1. Autoencoders and variational autoencoders
2. Generative Adversarial Networks (GANs) basics
3. Image generation and manipulation
4. Latent space exploration
5. **Build**: Simple image generator""",
    
    85: """### Day 85: Model Optimization & Deployment
1. Model quantization and pruning
2. TensorFlow Lite and mobile deployment
3. Model serving with TensorFlow Serving
4. Performance monitoring in production
5. **Build**: Deployed ML model API""",
    
    86: """### Day 86: Transformer Architecture
1. Attention mechanism deep dive
2. Transformer architecture components
3. BERT and GPT model families
4. Positional encoding and self-attention
5. **Build**: Text summarization with transformers""",
    
    87: """### Day 87: Large Language Models (LLMs)
1. Understanding GPT architecture
2. Fine-tuning pre-trained models
3. Prompt engineering techniques
4. LLM limitations and biases
5. **Build**: Custom chatbot with fine-tuned model""",
    
    88: """### Day 88: Computer Vision Advanced Topics
1. Object detection and segmentation
2. Facial recognition systems
3. Image style transfer
4. Video analysis and tracking
5. **Build**: Real-time object detection system""",
    
    89: """### Day 89: Reinforcement Learning
1. Q-learning and policy gradients
2. Deep Q-Networks (DQN)
3. Actor-Critic methods
4. Multi-agent environments
5. **Build**: Game-playing AI agent""",
    
    90: """### Day 90: AI Ethics & Bias
1. Algorithmic bias and fairness
2. Explainable AI (XAI) techniques
3. Privacy-preserving ML
4. Responsible AI development
5. **Build**: Bias detection and mitigation tool""",
    
    91: """### Day 91: MLOps & Production Systems
1. ML pipeline automation
2. Model versioning and registry
3. A/B testing for ML models
4. Monitoring and drift detection
5. **Build**: Complete MLOps pipeline""",
    
    92: """### Day 92: Advanced NLP Applications
1. Named Entity Recognition (NER)
2. Question answering systems
3. Dialogue systems and chatbots
4. Language translation models
5. **Build**: Multi-task NLP application""",
    
    93: """### Day 93: AI Research & Innovation
1. Reading and understanding research papers
2. Implementing cutting-edge techniques
3. Contributing to open-source AI projects
4. Staying current with AI trends
5. **Build**: Implementation of recent research paper""",
    
    94: """### Day 94: AI Specialization Choice
**Choose one specialization:**
1. **Computer Vision**: Advanced object detection, medical imaging
2. **NLP**: Advanced language models, multilingual systems
3. **Robotics**: Robot control, sensor fusion
4. **AI for Business**: Recommendation systems, fraud detection
5. **Build**: Advanced project in chosen specialization""",
    
    95: """### Day 95: Advanced Tools & Frameworks
1. PyTorch vs TensorFlow deep dive
2. Hugging Face Transformers library
3. Weights & Biases for experiment tracking
4. Docker for ML model deployment
5. **Build**: Professional-grade ML system""",
    
    96: """### Day 96: AI System Architecture
1. Scalable ML system design
2. Microservices for AI applications
3. Real-time inference systems
4. Edge AI and mobile deployment
5. **Build**: Scalable AI architecture""",
    
    97: """### Day 97: Research & Development
1. Conducting AI experiments
2. Hypothesis-driven development
3. A/B testing for AI systems
4. Publishing and presenting results
5. **Build**: Research-quality AI project""",
    
    98: """### Day 98: Industry Applications
1. AI in healthcare and finance
2. Autonomous systems and robotics
3. AI for climate and sustainability
4. Creative AI applications
5. **Build**: Industry-specific AI solution""",
    
    99: """### Day 99: AI Leadership & Strategy
1. Leading AI teams and projects
2. AI product management
3. Technical debt in AI systems
4. Building AI-first organizations
5. **Build**: AI strategy document and roadmap""",
    
    100: """### Day 100: AI Mastery Capstone
1. Portfolio project showcase
2. Code review and optimization
3. Documentation and presentation
4. Future learning path planning
5. **Build**: Comprehensive AI portfolio"""
}

# Function to create files with both prompts
def create_course_files(start_day, end_day):
    for day in range(start_day, end_day + 1):
        folder_name = f"Day_{day}"
        
        # Create folder if it doesn't exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Created folder: {folder_name}")
        
        # File 1: with prompt_1 (excludes number 5)
        file_name_1 = f"day_{day}_part1.md"
        file_path_1 = os.path.join(folder_name, file_name_1)
        
        # File 2: with prompt_2 (only number 5)
        file_name_2 = f"day_{day}_part2.md"
        file_path_2 = os.path.join(folder_name, file_name_2)
        
        try:
            # Get day content
            content = day_content.get(day, f"### Day {day}\n- Content for Day {day} to be added\n")
            
            # Create file 1 with prompt_1
            with open(file_path_1, 'w') as f:
                f.write(prompt_1 + "\n" + content)
            print(f"Created file: {file_path_1}")
            
            # Create file 2 with prompt_2
            with open(file_path_2, 'w') as f:
                f.write(prompt_2 + "\n" + content)
            print(f"Created file: {file_path_2}")
            
        except OSError as e:
            print(f"Error creating files for day {day}: {e}")

# Function to update existing files (like the original script)
def update_existing_files(start_day, end_day, prompt_choice=1):
    prompt = prompt_1 if prompt_choice == 1 else prompt_2
    
    for day in range(start_day, end_day + 1):
        folder_name = f"Day_{day}"
        file_name = f"day_{day}.md"
        file_path = os.path.join(folder_name, file_name)

        try:
            # Check if the file exists
            if os.path.exists(file_path):
                # Read existing content
                with open(file_path, 'r') as f:
                    existing_content = f.read()
                
                # Prepare new content: prompt + day-specific content + existing content (if any)
                new_content = prompt + "\n" + day_content.get(day, f"### Day {day}\n- Content for Day {day} to be added\n") + "\n\n" + existing_content
                
                # Write the updated content back to the file
                with open(file_path, 'w') as f:
                    f.write(new_content)
                print(f"Updated file: {file_path}")
            else:
                print(f"File not found: {file_path}")
        except OSError as e:
            print(f"Error updating file {file_path}: {e}")

# Main execution
if __name__ == "__main__":
    print("AI Mastery Course File Generator")
    print("=" * 50)
    
    choice = input("Choose an option:\n1. Create new files with both prompts (recommended)\n2. Update existing files with prompt 1\n3. Update existing files with prompt 2\nEnter choice (1/2/3): ")
    
    if choice == "1":
        print("\nCreating new files for Days 70-100 with both prompts...")
        create_course_files(70, 100)
        print("\nCompleted creating AI Mastery Course files!")
        print("Each day now has two files:")
        print("- day_X_part1.md (with prompt 1 - excludes number 5)")
        print("- day_X_part2.md (with prompt 2 - only number 5)")
        
    elif choice == "2":
        print("\nUpdating existing files for Days 70-100 with prompt 1...")
        update_existing_files(70, 100, prompt_choice=1)
        print("\nCompleted updating files with prompt 1!")
        
    elif choice == "3":
        print("\nUpdating existing files for Days 70-100 with prompt 2...")
        update_existing_files(70, 100, prompt_choice=2)
        print("\nCompleted updating files with prompt 2!")
        
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")