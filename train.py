"""
Main Training Script - Fine-tune on Full SAMSum Dataset
Based on successful unit test configuration
"""

import os
import json
import torch
from pathlib import Path
from typing import List, Dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')


class SummarizationFineTuner:
    """Fine-tune language models for dialogue summarization using LoRA"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        output_dir: str = "fine_tuned_model"
    ):
        """Initialize the fine-tuner"""
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
        
        # Load tokenizer
        print(f"\nLoading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"✓ Tokenizer loaded")
        
        # Load base model
        print(f"\nLoading base model from {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Same as unit test
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            use_cache=False
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        print(f"✓ Model loaded ({self.model.num_parameters():,} parameters)")
        
    def load_data(self, filepath: str) -> List[Dict]:
        """Load training data from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"\nLoaded {len(data)} training examples from {filepath}")
        return data
    
    def format_instruction(self, example: Dict) -> str:
        """Format example as instruction-response pair"""
        instruction = example['instruction']
        response = example['response']
        
        formatted = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        
        return formatted
    
    def prepare_dataset(self, data: List[Dict], max_length: int = 512) -> Dataset:
        """Prepare dataset for training"""
        print(f"\nPreparing dataset...")
        
        formatted_texts = [self.format_instruction(example) for example in data]
        
        print(f"Tokenizing {len(formatted_texts)} examples...")
        encodings = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        dataset = Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": encodings["input_ids"].clone()
        })
        
        print(f"✓ Dataset prepared with {len(dataset)} examples")
        return dataset
    
    def setup_lora(self, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05):
        """Setup LoRA configuration - same as unit test"""
        print(f"\nSetting up LoRA configuration...")
        print(f"  Rank (r): {r}")
        print(f"  Alpha: {lora_alpha}")
        print(f"  Dropout: {lora_dropout}")
        
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        print(f"  Target modules: {target_modules}")
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        print(f"\n✓ LoRA model created!")
        self.model.print_trainable_parameters()
        
        return self.model
    
    def train(
        self,
        train_dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 2,
        learning_rate: float = 5e-5,  # Same as unit test
        warmup_steps: int = 100,
        logging_steps: int = 50,
        save_steps: int = 500,
        gradient_accumulation_steps: int = 4  # Same as unit test
    ):
        """Train the model with LoRA"""
        print(f"\n{'='*60}")
        print("STARTING TRAINING")
        print(f"{'='*60}")
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Same configuration as unit test
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=2,
            fp16=False,  # Same as unit test
            bf16=False,
            optim="adamw_torch",
            lr_scheduler_type="linear",  # Same as unit test
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=False,
            max_grad_norm=1.0,  # Same as unit test
            dataloader_num_workers=0,
            group_by_length=False,
            logging_first_step=True
        )
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Max grad norm: 1.0")
        total_steps = (len(train_dataset) * num_epochs) // (batch_size * gradient_accumulation_steps)
        print(f"  Total steps: ~{total_steps}")
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        print(f"\nStarting training...")
        print(f"Expected duration: 30-40 minutes")
        print(f"{'='*60}")
        
        train_result = trainer.train()
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"Final training loss: {train_result.training_loss:.4f}")
        
        # Save final model
        print(f"\nSaving fine-tuned model to {self.output_dir}...")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"✓ Model saved successfully!")
        
        return train_result


def main(
    data_path: str = "data/train_data.json",
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: str = "fine_tuned_model",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    lora_r: int = 8,
    max_length: int = 512
):
    """Main training pipeline"""
    print(f"\n{'='*60}")
    print("DIALOGUE SUMMARIZATION MODEL FINE-TUNING")
    print(f"SAMSum Dataset - Full Training")
    print(f"{'='*60}\n")
    
    fine_tuner = SummarizationFineTuner(
        model_name=model_name,
        output_dir=output_dir
    )
    
    train_data = fine_tuner.load_data(data_path)
    train_dataset = fine_tuner.prepare_dataset(train_data, max_length=max_length)
    
    fine_tuner.setup_lora(r=lora_r)
    
    train_result = fine_tuner.train(
        train_dataset=train_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    print(f"\n{'='*60}")
    print("FINE-TUNING COMPLETE!")
    print(f"{'='*60}")
    print(f"Model saved to: {output_dir}")
    print(f"\nNext step - Evaluate with:")
    print(f"  CUDA_VISIBLE_DEVICES=3 python evaluate.py")
    print(f"{'='*60}\n")
    
    return fine_tuner, train_result


if __name__ == "__main__":
    import time
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    start_time = time.time()
    
    print(f"\n{'#'*60}")
    print("STARTING FULL TRAINING")
    print(f"{'#'*60}\n")
    
    try:
        fine_tuner, result = main(
            data_path="data/train_data.json",
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            output_dir="fine_tuned_model",
            num_epochs=3,
            batch_size=2,
            learning_rate=5e-5,  # Same as unit test
            lora_r=8  # Same as unit test
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'#'*60}")
        print(f"TRAINING COMPLETED IN {duration/60:.1f} MINUTES")
        print(f"{'#'*60}\n")
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED WITH ERROR:")
        print(f"{e}")
        import traceback
        traceback.print_exc()