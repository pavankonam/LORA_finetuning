"""
Data Preparation Script with SAMSum Dataset
Uses dialogue summarization dataset (15K examples)
Much faster than Sentiment140!
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from datasets import load_dataset
import random

random.seed(42)


class SamSumDataProcessor:
    """Process SAMSum dataset for dialogue summarization fine-tuning"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_samsum(self):
        """Load SAMSum dataset from HuggingFace"""
        print(f"\n{'='*60}")
        print("LOADING SAMSUM DATASET")
        print(f"{'='*60}\n")
        
        print("Downloading from HuggingFace...")
        dataset = load_dataset("knkarthick/samsum")
        
        print(f"✓ Loaded successfully")
        print(f"  Train: {len(dataset['train'])} examples")
        print(f"  Test: {len(dataset['test'])} examples")
        print(f"  Validation: {len(dataset['validation'])} examples")
        
        return dataset
    
    def create_instruction_format(self, dialogue: str, summary: str) -> Dict[str, str]:
        """
        Convert dialogue to instruction-response format
        
        Args:
            dialogue: Conversation text
            summary: Summary of the conversation
        
        Returns:
            Dictionary with instruction and response
        """
        instruction = f"Summarize the following conversation:\n\n{dialogue}"
        
        response = f"Summary: {summary}"
        
        return {
            "instruction": instruction,
            "response": response,
            "dialogue": dialogue,
            "summary": summary
        }
    
    def prepare_datasets(
        self,
        train_size: int = None,
        test_size: int = 100
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Prepare training and test datasets
        
        Args:
            train_size: Number of training examples (None = use all)
            test_size: Number of test examples
        
        Returns:
            Tuple of (train_data, test_data)
        """
        # Load dataset
        dataset = self.load_samsum()
        
        print(f"\n{'='*60}")
        print("CREATING TRAIN/TEST SPLIT")
        print(f"{'='*60}\n")
        
        # Use train split for training
        train_examples = dataset['train']
        if train_size and train_size < len(train_examples):
            # Sample if train_size specified
            indices = random.sample(range(len(train_examples)), train_size)
            train_examples = train_examples.select(indices)
        
        # Use test split for testing
        test_examples = dataset['test']
        if test_size and test_size < len(test_examples):
            indices = random.sample(range(len(test_examples)), test_size)
            test_examples = test_examples.select(indices)
        
        print(f"Training set: {len(train_examples)} examples")
        print(f"Test set: {len(test_examples)} examples")
        
        # Convert to instruction format
        print(f"\nConverting to instruction format...")
        train_data = [
            self.create_instruction_format(ex['dialogue'], ex['summary'])
            for ex in train_examples
        ]
        
        test_data = [
            self.create_instruction_format(ex['dialogue'], ex['summary'])
            for ex in test_examples
        ]
        
        print(f"✓ Conversion complete")
        
        return train_data, test_data
    
    def save_datasets(
        self,
        train_data: List[Dict],
        test_data: List[Dict],
        train_path: str = "data/train_data.json",
        test_path: str = "data/test_questions.json"
    ):
        """Save datasets to JSON files"""
        print(f"\n{'='*60}")
        print("SAVING DATASETS")
        print(f"{'='*60}\n")
        
        # Save training data
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved training data: {train_path}")
        print(f"  Size: {os.path.getsize(train_path) / 1024 / 1024:.2f} MB")
        
        # Save test data
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved test data: {test_path}")
        print(f"  Size: {os.path.getsize(test_path) / 1024:.2f} KB")
        
        # Also create small unit test dataset
        unit_test_data = train_data[:100]  # First 100 examples
        unit_test_path = "data/unit_test_data.json"
        with open(unit_test_path, 'w', encoding='utf-8') as f:
            json.dump(unit_test_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved unit test data: {unit_test_path}")
        print(f"  Size: {os.path.getsize(unit_test_path) / 1024:.2f} KB")


def main(train_size: int = None, test_size: int = 100):
    """
    Main function to prepare all datasets
    
    Args:
        train_size: Number of training examples (None = use all ~14.7K)
        test_size: Number of test examples (default: 100)
    """
    print(f"\n{'#'*60}")
    print("SAMSUM DATASET PREPARATION")
    print(f"{'#'*60}\n")
    
    processor = SamSumDataProcessor()
    
    # Prepare datasets
    train_data, test_data = processor.prepare_datasets(
        train_size=train_size,
        test_size=test_size
    )
    
    # Save datasets
    processor.save_datasets(train_data, test_data)
    
    print(f"\n{'='*60}")
    print("DATA PREPARATION COMPLETE!")
    print(f"{'='*60}")
    print(f"\nDatasets ready:")
    print(f"  - Training: {len(train_data):,} examples")
    print(f"  - Test: {len(test_data):,} examples")
    print(f"  - Unit test: 100 examples")
    print(f"\nYou can now run:")
    print(f"  python unit_test.py")
    print(f"  python train.py")
    print(f"\n{'#'*60}\n")


if __name__ == "__main__":
    # Use all training data (~14,732 examples)
    # Or specify a number like: main(train_size=5000, test_size=100)
    main(train_size=None, test_size=100)