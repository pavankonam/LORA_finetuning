"""
Evaluation Script - Compare Base Model vs Fine-Tuned Model
Tests on held-out SAMSum test set
"""

import os
import json
import torch
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')


def load_test_data(filepath: str = "data/test_questions.json") -> List[Dict]:
    """Load test examples"""
    with open(filepath, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} test examples from {filepath}")
    return examples


def format_instruction(example, tokenizer):
    """Format example as instruction-response pair"""
    instruction = example['instruction']
    response = example['response']
    
    formatted = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    
    return formatted


def prepare_dataset(data, tokenizer, max_length=512):
    """Prepare dataset for evaluation"""
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
    
    print(f"Evaluating on {len(eval_dataset)} examples...")
    
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


def generate_summary(model, tokenizer, dialogue: str, device, max_new_tokens: int = 100):
    """Generate summary from model"""
    prompt = f"Summarize the following conversation:\n\n{dialogue}"
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract summary
    if "<|im_start|>assistant" in response:
        summary = response.split("<|im_start|>assistant")[-1].strip()
        if "<|im_end|>" in summary:
            summary = summary.split("<|im_end|>")[0].strip()
    else:
        summary = response
    
    model.train()
    return summary


class SummarizationEvaluator:
    """Evaluate and compare base and fine-tuned models"""
    
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        finetuned_model_path: str = "fine_tuned_model"
    ):
        """Initialize evaluator with both models"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}\n")
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Load base model
        print(f"Loading BASE model: {base_model_name}")
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,  # Same as training
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.device == "cpu":
            self.base_model = self.base_model.to(self.device)
        
        print("✓ Base model loaded\n")
        
        # Load fine-tuned model
        print(f"Loading FINE-TUNED model from: {finetuned_model_path}")
        self.finetuned_tokenizer = AutoTokenizer.from_pretrained(
            finetuned_model_path,
            trust_remote_code=True
        )
        if self.finetuned_tokenizer.pad_token is None:
            self.finetuned_tokenizer.pad_token = self.finetuned_tokenizer.eos_token
        
        base_for_peft = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,  # Same as training
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        self.finetuned_model = PeftModel.from_pretrained(
            base_for_peft,
            finetuned_model_path
        )
        
        if self.device == "cpu":
            self.finetuned_model = self.finetuned_model.to(self.device)
        
        print("✓ Fine-tuned model loaded\n")
    
    def evaluate_on_test_set(self, test_data: List[Dict]) -> Dict:
        """Evaluate both models on test set"""
        print(f"\n{'='*60}")
        print(f"EVALUATING MODELS ON {len(test_data)} TEST EXAMPLES")
        print(f"{'='*60}\n")
        
        # Prepare test dataset
        test_dataset = prepare_dataset(test_data, self.base_tokenizer, max_length=512)
        
        # Evaluate base model
        print("Evaluating BASE model...")
        base_loss = evaluate_loss(self.base_model, test_dataset, self.device)
        print(f"✓ Base model loss: {base_loss:.4f}\n")
        
        # Evaluate fine-tuned model
        print("Evaluating FINE-TUNED model...")
        ft_loss = evaluate_loss(self.finetuned_model, test_dataset, self.device)
        print(f"✓ Fine-tuned model loss: {ft_loss:.4f}\n")
        
        results = {
            "base_model_loss": base_loss,
            "finetuned_model_loss": ft_loss,
            "improvement": base_loss - ft_loss,
            "improvement_pct": ((base_loss - ft_loss) / base_loss * 100) if base_loss > 0 else 0
        }
        
        print(f"{'='*60}")
        print("EVALUATION COMPLETE!")
        print(f"{'='*60}\n")
        
        return results
    
    def show_example_comparisons(self, test_data: List[Dict], num_examples: int = 5):
        """Show example summaries from both models"""
        print(f"\n{'='*60}")
        print("EXAMPLE COMPARISONS")
        print(f"{'='*60}\n")
        
        for i in range(min(num_examples, len(test_data))):
            example = test_data[i]
            dialogue = example['dialogue']
            true_summary = example['summary']
            
            print(f"\nExample {i+1}:")
            print(f"Dialogue: {dialogue[:150]}...")
            print(f"\nTrue Summary:\n{true_summary}")
            
            # Base model
            print(f"\nBase Model Summary:")
            base_summary = generate_summary(
                self.base_model,
                self.base_tokenizer,
                dialogue,
                self.device
            )
            print(base_summary[:200])
            
            # Fine-tuned model
            print(f"\nFine-Tuned Model Summary:")
            ft_summary = generate_summary(
                self.finetuned_model,
                self.finetuned_tokenizer,
                dialogue,
                self.device
            )
            print(ft_summary[:200])
            
            print(f"\n{'-'*60}")
    
    def save_results(self, results: Dict, output_file: str = "evaluation_results.json"):
        """Save results to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to {output_file}")


def main(
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    finetuned_model_path: str = "fine_tuned_model",
    test_path: str = "data/test_questions.json",
    output_file: str = "evaluation_results.json"
):
    """Main evaluation pipeline"""
    print(f"\n{'='*60}")
    print("SUMMARIZATION MODEL EVALUATION")
    print(f"{'='*60}\n")
    
    # Load test data
    test_data = load_test_data(test_path)
    
    # Initialize evaluator
    evaluator = SummarizationEvaluator(
        base_model_name=base_model_name,
        finetuned_model_path=finetuned_model_path
    )
    
    # Evaluate on test set
    results = evaluator.evaluate_on_test_set(test_data)
    
    # Print summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Base model loss:       {results['base_model_loss']:.4f}")
    print(f"Fine-tuned model loss: {results['finetuned_model_loss']:.4f}")
    print(f"Improvement:           {results['improvement']:.4f} ({results['improvement_pct']:.1f}%)")
    
    if results['improvement'] > 0:
        print(f"\n✅ SUCCESS - Fine-tuning improved the model!")
    else:
        print(f"\n⚠️  WARNING - No improvement detected")
    
    print(f"{'='*60}")
    
    # Show example comparisons
    evaluator.show_example_comparisons(test_data, num_examples=3)
    
    # Save results
    evaluator.save_results(results, output_file)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    import time
    
    start_time = time.time()
    
    print(f"\n{'#'*60}")
    print("STARTING EVALUATION")
    print(f"{'#'*60}\n")
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    try:
        results = main(
            base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
            finetuned_model_path="fine_tuned_model",
            test_path="data/test_questions.json",
            output_file="evaluation_results.json"
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'#'*60}")
        print(f"EVALUATION COMPLETED IN {duration/60:.1f} MINUTES")
        print(f"{'#'*60}\n")
        
    except Exception as e:
        print(f"\n❌ EVALUATION FAILED WITH ERROR:")
        print(f"{e}")
        import traceback
        traceback.print_exc()