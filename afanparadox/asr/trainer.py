"""
Training pipeline for ASR (Speech Recognition)
"""

import torch
import torch.nn as nn
from torch.utils_data import DataLoader
from tqdm import tqdm
import os


class ASRTrainer:
    """
    Trainer for Wav2Vec2-based ASR models
    """
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        lr=1e-4,
        batch_size=8,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if val_dataset:
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
    def train(self, epochs=10, save_path="checkpoints/asr"):
        os.makedirs(save_path, exist_ok=True)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in pbar:
                # In a real pipeline, the dataset would return input_values and labels
                input_values = batch["input_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(input_values, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
                
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            torch.save(self.model.state_dict(), os.path.join(save_path, f"checkpoint_e{epoch}.pt"))
            
            if self.val_dataset:
                self.validate()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        for batch in self.val_loader:
            input_values = batch["input_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            outputs = self.model(input_values, labels=labels)
            total_loss += outputs.loss.item()
            
        print(f"Validation Loss: {total_loss / len(self.val_loader):.4f}")
        self.model.train()
