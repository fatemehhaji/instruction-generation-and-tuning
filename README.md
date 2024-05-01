# Instruction-Based Dataset Generation and LLM Fine-Tuning

## Setup

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/fatemehhaji/instruction_generation_and_tuning.git
2. Create and activate a Virtual Environment.
3. Install Dependencies
   
## Objective
This project explores the process of generating an instruction-based dataset. We fine-tuned a pre-trained model using this new dataset and compared its performance with a version fine-tuned on both this dataset and Alpaca. Additionally, we tested the model's behavior using general-purpose instructions.
## Tasks Overview
1. **Instruction-Based Dataset Generation**: We transformed a non-instruction-based dataset into an instruction-based format using GPT-4.
2. **Model Fine-Tuning**: We fine-tuned a pre-trained Large Language Model (LLM) using both the newly created instruction-based dataset and a combination of this dataset with Alpaca.
3. **Comparison and Analysis**: We evaluated the performance of the fine-tuned models and the original pre-trained model, focusing on their ability to generate relevant responses based on new instructions.

## Dataset and Model Details
- **Original Dataset Source**: The process of adding GPT-4 generated instructions to the data is in the code, and since the dataset is from Hugging Face, everything is included.
- **Models Used**:
  - **Original Pre-trained Model**: [mistralai/Mistral-7B-Instruct-v0.2](#)
  - **Fine-tuned Models**: 
    - Fine-tuned on instruction-based dataset. (Mistral-7B-Sentiment-Tuned)
    - Fine-tuned on combined dataset. (Mistral-7B-Instruct-Sentiment-Tuned)

## Results
### Twitter Sentiment Analysis Results
Three models were assessed based on precision, recall, F1 score, and accuracy:
- **Mistral-7B-Instruct-Sentiment-Tuned**
- **Mistral-7B-Sentiment-Tuned**
- **Mistral-7B-Instruct-v0.2**

### Performance Metrics
| Model                                 | Precision | Recall | F1 Score | Accuracy |
|---------------------------------------|-----------|--------|----------|----------|
| **Mistral-7B-Instruct-Sentiment-Tuned** | 0.9048    | 0.8261 | 0.8636   | 0.8800   |
| **Mistral-7B-Sentiment-Tuned**         | 0.9048    | 0.8261 | 0.8636   | 0.8800   |
| **Mistral-7B-Instruct-v0.2**           | 0.8000    | 0.8696 | 0.8333   | 0.8400   |

### Alpaca Dataset Results
| Model                              | Perplexity | BLEU    | ROUGE-L | BERTScore |
|------------------------------------|------------|---------|---------|-----------|
| **Mistral-7B-Instruct-v0.2**       | 46.0914    | 0.23497 | 0.4518  | 0.9072    |
| **Mistral-7B-Instruct-Sentiment-Tuned** | 6.9803 | 0.06092 | 0.1918  | 0.8668    |
| **Mistral-7B-Sentiment-Tuned**     | 11.6970    | 0.02733 | 0.0652  | 0.2647    |

Fine-tuning with the sentiment dataset significantly enhanced precision and accuracy, improving the model's ability to detect sentiments correctly.

### Out-of-Sample Instructions and Comparisons
We generated 10 completely out-of-sample instructions to test the models' generalization ability. These instructions were unrelated to the original training data.
The responses are in the `results/responses.json` file.

- **Original Pre-trained Model**: Showed strong generalization, providing contextually relevant responses for most instructions.
- **Mistral-7B-Instruct-Sentiment-Tuned**: While able to handle some of the out-of-sample instructions well, responses were sometimes influenced by fine-tuning.
- **Mistral-7B-Sentiment-Tuned**: Exhibited noticeable degradation in generalization, producing more specific responses that often failed to appropriately address the instruction.

### Further Steps
To enhance generalizability and performance:
- **Multi-Task Learning**: Integrate various training tasks to balance specialization with generalization.
- **Regularization Techniques**: Implement dropout and weight decay.
- **Continual Learning**: Incremental fine-tuning while retaining prior knowledge.

## References
- Data sources: [Original Data Source](https://huggingface.co/datasets/carblacac/twitter-sentiment-analysis)

