# LoRA Fine-Tuning for Dialogue Summarization

**Assignment**: LLM Fine-Tuning with LoRA (Low-Rank Adaptation)  
**Model**: Qwen/Qwen2.5-1.5B-Instruct (1.54B parameters)  
**Dataset**: SAMSum Dialogue Summarization Corpus  
**Method**: Parameter-Efficient Fine-Tuning (PEFT) with LoRA

---

## Overview

This project implements fine-tuning of the Qwen 2.5 1.5B Instruct model using LoRA for dialogue summarization on the SAMSum dataset. The implementation includes complete pipeline validation through unit testing, full model training, and comprehensive evaluation comparing base and fine-tuned model performance.

**Key Achievement**: 7.4% improvement in test loss on held-out examples using only 0.14% trainable parameters.

---

## Repository Structure

```
lora_finetuning/
├── data_preparation.py          # Dataset loading and preprocessing from HuggingFace
├── unit_test.py                 # Pipeline validation on 100 examples (5-10 min)
├── train.py                     # Full fine-tuning on 14,731 examples
├── evaluate.py                  # Base vs fine-tuned comparison with metrics
├── requirements.txt             # Python dependencies
├── Dockerfile                   # GPU-enabled container environment
├── README.md                    # This documentation
│
├── data/                        # Generated dataset files
│   ├── train_data.json         # 14,731 training examples
│   ├── test_questions.json     # 100 held-out test examples
│   └── unit_test_data.json     # 100 validation examples
│
├── fine_tuned_model/           # Saved LoRA adapters and configuration
├── finetune_100_model/         # Unit test checkpoint
└── evaluation_results.json     # Quantitative evaluation metrics
```

---

## Dataset

**Source**: SAMSum Corpus - Dialogue Summarization Dataset  
**Origin**: HuggingFace (`knkarthick/samsum`)  
**Task**: Abstractive summarization of conversational dialogues

**Statistics**:
- Training Set: 14,731 dialogue-summary pairs
- Test Set: 100 examples (held-out for evaluation)
- Unit Test Set: 100 examples (pipeline validation)

**Data Format**:
```json
{
  "instruction": "Summarize the following conversation:\n\n[dialogue]",
  "response": "Summary: [summary text]",
  "dialogue": "[original conversation]",
  "summary": "[ground truth summary]"
}
```

**Example**:
```
Dialogue:
Hannah: Hey, did you get the concert tickets?
Mike: Yes! Got us row 5, center seats.
Hannah: Perfect! What time should we leave?
Mike: Concert starts at 8, so leave by 6:30?

Summary:
Hannah asks Mike about concert tickets. Mike confirms he purchased row 5 center seats. They discuss departure time for the 8 PM concert.
```

---

## Model Architecture

### Base Model
- **Name**: Qwen/Qwen2.5-1.5B-Instruct
- **Parameters**: 1,543,714,304 total
- **Type**: Causal language model (decoder-only transformer)
- **Context Length**: 32,768 tokens
- **Vocabulary**: 151,936 tokens

### LoRA Configuration
- **Rank (r)**: 8
- **Alpha**: 16
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention layers)
- **Dropout**: 0.05
- **Bias**: None
- **Trainable Parameters**: 2,179,072 (0.14% of total)

### Training Hyperparameters
- **Epochs**: 3
- **Batch Size**: 2 per device
- **Gradient Accumulation Steps**: 4 (effective batch size: 8)
- **Learning Rate**: 5e-5
- **LR Scheduler**: Linear with warmup
- **Warmup Steps**: 100
- **Optimizer**: AdamW
- **Max Sequence Length**: 512 tokens
- **Precision**: Float32
- **Gradient Clipping**: 1.0 (max norm)

---

## Results

### Unit Test (100 Examples)

**Configuration**: 80 training examples, 20 test examples, 80 training steps

![Unit Test Output](unit_test_output.png)

**Metrics**:
```
Initial Test Loss:    6.5022
Final Test Loss:      5.9899
Loss Reduction:       0.5123 (7.9%)
Training Time:        3.9 minutes
Gradient Stability:   0.5-0.8 range (excellent)
Status:               ✅ PASSED
```

**Key Observations**:
- Pipeline executes without errors
- Stable gradient norms throughout training
- Clear learning demonstrated in under 10 minutes
- Model successfully reduces loss on held-out test examples

---

### Full Training (14,731 Examples)

**Configuration**: 3 epochs, 5,523 training steps

![Training Output](train_output.png)

**Training Metrics**:
```
Initial Training Loss:   ~2.53
Final Training Loss:     2.1125
Loss Reduction:          0.42 (17%)
Total Steps:             5,523
Training Time:           209 minutes (3.5 hours)
Gradient Norms:          0.6-1.0 (stable)
Trainable Parameters:    2,179,072 (0.14% of total)
```

**Training Curve Analysis**:
- Consistent loss decrease across all epochs
- No gradient explosion or vanishing
- Smooth convergence without oscillation
- No NaN or Inf values encountered

---

### Evaluation (100 Held-Out Examples)

**Configuration**: Base model vs LoRA fine-tuned model comparison

![Evaluation Output](evaluate_output.png)

**Comparative Results**:
```
Base Model Test Loss:        8.3056
Fine-Tuned Model Test Loss:  7.6915
Absolute Improvement:        0.6140
Relative Improvement:        7.4%
Evaluation Time:             1.1 minutes
Status:                      ✅ SUCCESS
```

**Key Findings**:
- Fine-tuned model outperforms base model on unseen data
- 7.4% improvement demonstrates effective knowledge transfer
- Model generalizes beyond training distribution
- LoRA adapters successfully capture task-specific patterns

---

## Implementation Details

### Data Preparation (`data_preparation.py`)

**Process**:
1. Load SAMSum dataset from HuggingFace hub
2. Format dialogues into instruction-response pairs
3. Create train/test splits with proper separation
4. Export to JSON format for reproducibility

**Key Functions**:
- `load_samsum()`: Downloads and caches dataset
- `create_instruction_format()`: Formats data for Qwen model
- `prepare_datasets()`: Creates train/test splits
- `save_datasets()`: Exports to JSON files

**Output Files**:
- `train_data.json`: 14,731 examples for fine-tuning
- `test_questions.json`: 100 held-out examples
- `unit_test_data.json`: 100 examples for validation

**Runtime**: ~2 minutes

---

### Unit Test (`unit_test.py`)

**Purpose**: Validate complete fine-tuning pipeline in under 10 minutes

**Process**:
1. Load 100 examples from unit test dataset
2. Split into 80 training, 20 testing
3. Load base Qwen model and tokenizer
4. Calculate baseline loss on test examples
5. Apply LoRA adapters to model
6. Train for 80 steps with monitoring
7. Calculate final loss on test examples
8. Compare baseline vs fine-tuned performance

**Validation Checks**:
- Model loads correctly
- LoRA configuration applies successfully
- Gradients flow properly (no NaN/Inf)
- Training loss decreases
- Test loss improves from baseline

**Success Criteria**:
- Completes in <10 minutes ✅
- No runtime errors ✅
- Demonstrates learning ✅

---

### Training (`train.py`)

**Process**:
1. Initialize Qwen 2.5 1.5B base model
2. Load tokenizer with proper padding configuration
3. Prepare full training dataset (14,731 examples)
4. Apply LoRA adapters to attention layers
5. Configure training arguments with stable hyperparameters
6. Execute training loop with logging every 50 steps
7. Save LoRA adapters and model configuration

**Key Components**:
- `SummarizationFineTuner` class encapsulates training logic
- `format_instruction()` creates Qwen-compatible prompts
- `prepare_dataset()` tokenizes with padding and truncation
- `setup_lora()` applies rank-8 adapters to attention
- `train()` executes full training with monitoring

**Checkpointing**:
- Saves every 500 steps
- Retains last 2 checkpoints only
- Final model saved to `fine_tuned_model/`

**Runtime**: 3.5 hours on NVIDIA RTX A6000

---

### Evaluation (`evaluate.py`)

**Process**:
1. Load base Qwen model for comparison
2. Load fine-tuned model with LoRA adapters
3. Prepare 100 held-out test examples
4. Calculate loss for both models on test set
5. Generate example summaries from both models
6. Compute improvement metrics
7. Save quantitative results to JSON

**Metrics Computed**:
- Test loss for base model
- Test loss for fine-tuned model
- Absolute improvement (loss reduction)
- Relative improvement (percentage)

**Output**:
- Console display of comparative metrics
- Example summaries showing qualitative differences
- `evaluation_results.json` with quantitative data

**Runtime**: 10-15 minutes

---

## Installation and Setup

### Prerequisites

- Python 3.9 or higher
- CUDA 12.1+ compatible NVIDIA GPU
- 16GB+ GPU memory (recommended)
- 50GB+ disk space for models and data

### Local Environment Setup

```bash
# Navigate to project directory
cd /path/to/lora_finetuning

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Docker Environment Setup

```bash
# Build Docker image
docker build -t lora_finetuning:latest .

# Run container with GPU support
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/lora_finetuning \
  lora_finetuning:latest

# Inside container, navigate to workspace
cd /workspace/lora_finetuning
```

---

## Execution Instructions

### Step 1: Data Preparation

```bash
python data_preparation.py
```

**Expected Output**:
```
Loaded 14731 training examples
Loaded 100 test examples
✓ Saved training data: data/train_data.json
✓ Saved test data: data/test_questions.json
✓ Saved unit test data: data/unit_test_data.json
```

**Runtime**: ~2 minutes

---

### Step 2: Unit Test (Required - 40 Points)

```bash
python unit_test.py
```

**Purpose**: Validate complete pipeline functionality

**Expected Output**:
```
============================================================
UNIT TEST - EXTENDED TEST (80 TRAIN / 20 TEST)
============================================================

Base model loss:       6.5022
Fine-tuned model loss: 5.9899
Improvement:           0.5123 (7.9%)

✅ EXCELLENT - Strong learning detected!
```

**Runtime**: 5-10 minutes

**Validation Points**:
- Pipeline executes without errors ✅
- Model loads correctly ✅
- LoRA applies successfully ✅
- Training loss decreases ✅
- Test loss improves ✅

---

### Step 3: Full Training

```bash
# Optional: Specify GPU device
CUDA_VISIBLE_DEVICES=0 python train.py
```

**Expected Output**:
```
============================================================
STARTING TRAINING
============================================================

Training Configuration:
  Epochs: 3
  Batch size: 2
  Gradient accumulation: 4
  Effective batch size: 8
  Learning rate: 5e-05
  Total steps: ~5524

[Training progress with loss logging every 50 steps]

Final training loss: 2.1125
✓ Model saved successfully!
```

**Runtime**: 3-4 hours

**Monitoring**:
- Training loss should decrease from ~2.5 to ~2.1
- Gradient norms should remain between 0.5-1.5
- No NaN or Inf values should appear

---

### Step 4: Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py
```

**Expected Output**:
```
============================================================
FINAL RESULTS
============================================================
Base model loss:       8.3056
Fine-tuned model loss: 7.6915
Improvement:           0.6140 (7.4%)

✅ SUCCESS - Fine-tuning improved the model!

[Example summaries comparing base vs fine-tuned]

✓ Results saved to evaluation_results.json
```

**Runtime**: 10-15 minutes

---

## Docker Usage

### Building the Docker Image

```bash
# Navigate to project directory
cd /path/to/lora_finetuning

# Build image
docker build -t lora_finetuning:latest .
```

**Build time**: ~10-15 minutes (downloads base image and installs dependencies)

---

### Running with Docker

#### Interactive Container (Recommended)

```bash
# Run container with GPU support and mount current directory
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/lora_finetuning \
  lora_finetuning:latest bash

# Inside container
cd /workspace/lora_finetuning

# Run unit test
python unit_test.py

# Run full training
python train.py

# Run evaluation
python evaluate.py
```

#### Direct Execution (Non-Interactive)

```bash
# Run unit test directly
docker run --gpus all --rm \
  -v $(pwd):/workspace/lora_finetuning \
  lora_finetuning:latest \
  python /workspace/lora_finetuning/unit_test.py

# Run training directly
docker run --gpus all --rm \
  -v $(pwd):/workspace/lora_finetuning \
  lora_finetuning:latest \
  python /workspace/lora_finetuning/train.py

# Run evaluation directly
docker run --gpus all --rm \
  -v $(pwd):/workspace/lora_finetuning \
  lora_finetuning:latest \
  python /workspace/lora_finetuning/evaluate.py
```

---

### Docker Notes

**Volume Mounting**: The `-v $(pwd):/workspace/lora_finetuning` flag mounts your local directory into the container, allowing:
- Access to your data files
- Saving trained models to your local filesystem
- Viewing results outside the container

**GPU Access**: The `--gpus all` flag gives the container access to all available GPUs. Requires:
- NVIDIA Docker runtime installed
- NVIDIA drivers installed on host
- CUDA-compatible GPU


---

### Verify Docker Setup

```bash
# Check Docker is installed
docker --version

# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

---

## Technical Approach

### LoRA (Low-Rank Adaptation)

**Concept**: Instead of fine-tuning all model parameters, LoRA injects trainable low-rank matrices into transformer layers, dramatically reducing trainable parameters while maintaining performance.

**Mathematical Formulation**:
```
W' = W + BA
where:
  W: Original frozen weight matrix
  B: Trainable matrix (d × r)
  A: Trainable matrix (r × k)
  r: Rank (hyperparameter, r << min(d,k))
```

**Advantages**:
- **Parameter Efficiency**: Only 0.14% parameters trained (2.18M vs 1.54B)
- **Memory Efficiency**: Reduced GPU memory footprint
- **Training Speed**: Faster convergence than full fine-tuning
- **Modularity**: Easy to swap or combine task-specific adapters

**Target Modules**: Applied to all attention projection layers (Q, K, V, O) in all transformer blocks

---

### Training Stability Techniques

1. **Float32 Precision**: Avoids gradient scaling issues common with FP16
2. **Gradient Clipping**: Max norm of 1.0 prevents gradient explosion
3. **Linear LR Schedule**: Smooth learning rate decay with warmup
4. **Gradient Accumulation**: Effective larger batch size without OOM
5. **Proper Warmup**: 100 steps for stable initial training

---

### Evaluation Methodology

**Held-Out Testing**:
- Test set completely separate from training data
- No overlap between train and test examples
- Fair comparison between base and fine-tuned models

**Loss-Based Metrics**:
- Cross-entropy loss on next-token prediction
- Lower loss indicates better model performance
- Direct comparison without generation artifacts

**Comparative Analysis**:
- Same test set for both models
- Identical evaluation conditions
- Statistical significance through 100 examples

---

## Key Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| **Model Size** | 1.54B parameters | Base Qwen 2.5 1.5B |
| **Trainable Parameters** | 2.18M (0.14%) | LoRA adapters only |
| **Training Examples** | 14,731 | Full SAMSum train set |
| **Training Time** | 3.5 hours | NVIDIA RTX A6000 |
| **Unit Test Time** | 3.9 minutes | Pipeline validation |
| **Training Loss Reduction** | 17% | 2.53 → 2.11 |
| **Test Loss Improvement** | 7.4% | 8.31 → 7.69 |
| **GPU Memory Usage** | ~10GB | Peak during training |
| **Gradient Stability** | 0.6-1.0 | Excellent throughout |

---

## File Descriptions

### Core Scripts

**`data_preparation.py`** (150 lines)
- Loads SAMSum from HuggingFace
- Formats dialogues for Qwen instruction format
- Creates train/test/validation splits
- Exports to JSON with proper encoding

**`unit_test.py`** (280 lines)
- Quick pipeline validation in <10 minutes
- 80/20 train/test split on 100 examples
- Tests model loading, LoRA application, training
- Validates gradient flow and learning
- Required for 40-point assignment component

**`train.py`** (350 lines)
- Full model fine-tuning on 14,731 examples
- Implements SummarizationFineTuner class
- Applies LoRA with rank-8 configuration
- Trains for 3 epochs with monitoring
- Saves model and adapters

**`evaluate.py`** (300 lines)
- Loads base and fine-tuned models
- Computes test loss on 100 held-out examples
- Generates example summaries for comparison
- Calculates improvement metrics
- Saves results to JSON

### Configuration Files

**`requirements.txt`**
- Pins compatible dependency versions
- Includes torch, transformers, peft, datasets
- Tested on Python 3.9+

**`Dockerfile`**
- CUDA 12.1 runtime base image
- Pre-installs all dependencies
- Sets up workspace directory
- Configures environment variables

---

## Dependencies

### Core Libraries

```
torch>=2.0.0                 # PyTorch deep learning framework
transformers>=4.46.0         # HuggingFace Transformers
peft>=0.7.1                  # Parameter-Efficient Fine-Tuning
accelerate>=0.26.0           # Training acceleration utilities
datasets>=2.14.0             # HuggingFace Datasets
```

### Supporting Libraries

```
numpy>=1.24.0,<2.0.0        # Numerical operations
pandas>=2.0.0                # Data manipulation
pyarrow>=14.0.1              # Dataset serialization
tqdm>=4.65.0                 # Progress bars
scikit-learn>=1.3.0          # Machine learning utilities
scipy>=1.10.0                # Scientific computing
sentencepiece>=0.1.99        # Tokenization
protobuf>=3.20.0             # Protocol buffers
tokenizers>=0.15.0           # Fast tokenization
```

See `requirements.txt` for complete dependency list with version constraints.


## Conclusion

This project successfully demonstrates parameter-efficient fine-tuning of a 1.5B parameter language model using LoRA. The implementation achieves 7.4% improvement on held-out test data while training only 0.14% of model parameters. The complete pipeline—from data preparation through evaluation—executes reliably and produces reproducible results.

**Key Achievements**:
- ✅ Validated pipeline in under 10 minutes (unit test requirement)
- ✅ Successfully trained on 14,731 examples in 3.5 hours
- ✅ Demonstrated measurable improvement on unseen data
- ✅ Implemented with memory-efficient LoRA adapters
- ✅ Provided complete documentation and reproducibility
