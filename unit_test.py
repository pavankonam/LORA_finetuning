
import os
import json
import torch
from pathlib import Path
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


def load_unit_test_data(filepath: str = "data/unit_test_data.json"):
    """Load dataset for fine-tuning"""
    if not os.path.exists(filepath):
        print(f"Unit test data not found at {filepath}")
        print("Please run: python data_preparation.py first")
        exit(1)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {filepath}")
    return data


def format_instruction(example, tokenizer):
    """Format example as instruction-response pair for Qwen"""
    instruction = example['instruction']
    response = example['response']
    
    formatted = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    
    return formatted


def prepare_dataset(data, tokenizer, max_length=256):
    """Prepare dataset for training"""
    print(f"Tokenizing {len(data)} examples...")
    
    formatted_texts = [format_instruction(example, tokenizer) for example in data]
    
    encodings = tokenizer(
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
    
    print(f"Dataset prepared: {len(dataset)} examples")
    return dataset


def evaluate_loss(model, eval_dataset, device):
    """Calculate average loss on evaluation dataset"""
    model.eval()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for i in range(len(eval_dataset)):
            sample = eval_dataset[i]
            
            inputs = {
                'input_ids': torch.tensor(sample['input_ids']).unsqueeze(0).to(device),
                'attention_mask': torch.tensor(sample['attention_mask']).unsqueeze(0).to(device),
                'labels': torch.tensor(sample['labels']).unsqueeze(0).to(device)
            }
            
            outputs = model(**inputs)
            loss = outputs.loss
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                count += 1
    
    model.train()
    avg_loss = total_loss / count if count > 0 else float('inf')
    return avg_loss


def run_finetune_test(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    data_path: str = "data/unit_test_data.json",
    output_dir: str = "finetune_100_model",
    max_steps: int = 80  # suitable for 80 examples
):
    """
    Fine-tuning test on 100 examples (80 train / 20 test)
    """
    print(f"\n{'='*60}")
    print("EXTENDED UNIT TEST - 100 EXAMPLES (80 TRAIN / 20 TEST)")
    print(f"{'='*60}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    # Load data
    print(f"\n1. Loading dataset...")
    data = load_unit_test_data(data_path)
    print(f"   ✓ Loaded {len(data)} examples total")
    
    if len(data) < 100:
        print(f"⚠️  Warning: Only {len(data)} examples available. Using all.")
        split_point = int(len(data) * 0.8)
    else:
        split_point = 80
    
    train_data = data[:split_point]
    test_data = data[split_point:split_point + 20]
    
    print(f"   Using {len(train_data)} for training and {len(test_data)} for testing")
    
    # Load tokenizer
    print(f"\n2. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"   ✓ Tokenizer loaded")
    
    # Load model in float32 for stability
    print(f"\n3. Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        use_cache=False
    )
    
    print(f"   ✓ Model loaded ({model.num_parameters():,} parameters)")
    
    # Prepare datasets
    print(f"\n4. Preparing datasets...")
    train_dataset = prepare_dataset(train_data, tokenizer, max_length=256)
    test_dataset = prepare_dataset(test_data, tokenizer, max_length=256)
    print(f"   ✓ Datasets ready")
    
    # Evaluate base loss before training
    print(f"\n5. Evaluating base model loss...")
    base_loss = evaluate_loss(model, test_dataset, device)
    print(f"   ✓ Base model loss: {base_loss:.4f}")
    
    # Setup LoRA
    print(f"\n6. Setting up LoRA configuration...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    print(f"   ✓ LoRA applied")
    model.print_trainable_parameters()
    
    # Verify gradients
    print(f"\n7. Verifying gradients...")
    sample = train_dataset[0]
    inputs = {
        'input_ids': torch.tensor(sample['input_ids']).unsqueeze(0).to(device),
        'attention_mask': torch.tensor(sample['attention_mask']).unsqueeze(0).to(device),
        'labels': torch.tensor(sample['labels']).unsqueeze(0).to(device)
    }
    loss = model(**inputs).loss
    try:
        loss.backward()
        model.zero_grad()
        print(f"   ✓ Gradient check passed")
    except Exception as e:
        print(f"   ❌ Gradient check failed: {e}")
        raise
    
    # Training setup
    print(f"\n8. Configuring training...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=5,
        logging_steps=10,
        save_steps=max_steps,
        max_steps=max_steps,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=False,
        max_grad_norm=1.0,
        dataloader_num_workers=0,
        group_by_length=False,
        logging_first_step=True
    )
    
    print(f"   Config: {max_steps} steps, LR=5e-5, float32")
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    print(f"\n9. Starting training...")
    print(f"   Expected runtime: 4–7 minutes")
    print(f"   {'='*60}")
    
    train_result = trainer.train()
    
    print(f"   {'='*60}")
    print(f"   ✓ Training complete!")
    print(f"   Final training loss: {train_result.training_loss:.4f}")
    
    # Evaluate after fine-tuning
    print(f"\n10. Evaluating fine-tuned model...")
    finetuned_loss = evaluate_loss(model, test_dataset, device)
    print(f"    ✓ Fine-tuned model loss: {finetuned_loss:.4f}")
    
    # Save results
    print(f"\n11. Saving fine-tuned model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"    ✓ Saved to {output_dir}")
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Initial loss: {base_loss:.4f}")
    print(f"Final loss:   {finetuned_loss:.4f}")
    
    improvement = base_loss - finetuned_loss
    improvement_pct = (improvement / base_loss * 100) if base_loss > 0 else 0
    print(f"Loss reduction: {improvement:.4f} ({improvement_pct:.1f}%)")
    
    if improvement > 0.5:
        print("\n✅ EXCELLENT - Strong learning detected!")
    elif improvement > 0.1:
        print("\n✅ GOOD - Clear learning detected")
    elif improvement > 0.01:
        print("\n✅ MINIMAL - Some learning detected")
    elif improvement > 0:
        print("\n⚠️  WARNING - Minimal learning")
    else:
        print("\n❌ FAILED - No learning detected")
    print(f"{'='*60}")
    
    return model, train_result, base_loss, finetuned_loss


if __name__ == "__main__":
    import time
    
    start_time = time.time()
    
    print(f"\n{'#'*60}")
    print("STARTING 100-SAMPLE FINE-TUNING TEST")
    print(f"{'#'*60}")
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    try:
        model, result, base_loss, ft_loss = run_finetune_test(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            data_path="data/unit_test_data.json",
            output_dir="finetune_100_model",
            max_steps=80
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'#'*60}")
        print(f"COMPLETED IN {duration/60:.1f} MINUTES")
        print(f"Initial: {base_loss:.4f} → Final: {ft_loss:.4f}")
        print(f"Reduction: {base_loss - ft_loss:.4f}")
        print(f"{'#'*60}\n")
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED WITH ERROR:")
        print(f"{e}")
        import traceback
        traceback.print_exc()
