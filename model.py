"""
Transformer-based Binary Classifier with LoRA support
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType

class TransformerBinaryClassifier(nn.Module):
    """
    Transformer-based binary classifier for hate speech detection with LoRA support.
    """
    def __init__(self, model_name, dropout=0.1, use_lora=False, 
                 lora_r=8, lora_alpha=16, lora_dropout=0.05):
        """
        Initialize the binary classifier.
        
        Args:
            model_name (str): Name or path of pre-trained transformer model
            dropout (float): Dropout rate for regularization
            use_lora (bool): Whether to apply LoRA to the model
            lora_r (int): LoRA rank
            lora_alpha (int): LoRA alpha scaling factor
            lora_dropout (float): LoRA dropout rate
        """
        super(TransformerBinaryClassifier, self).__init__()
        
        # Load base encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size
        
        # Apply LoRA if requested
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,  # Changed from SEQ_CLS
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=["query", "value"],
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            self.encoder.print_trainable_parameters()
        
        # Classification head for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)  # Single output for binary classification
        )
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding
            
        Returns:
            dict: Dictionary containing logits
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        
        return {'logits': logits}

    
    def freeze_base_layers(self):
        """
        Freeze encoder parameters for feature extraction.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        frozen_params = sum(p.numel() for p in self.encoder.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Frozen {frozen_params:,} parameters out of {total_params:,} total parameters")
        print(f"Trainable parameters: {total_params - frozen_params:,}")
    
    def unfreeze_base_layers(self):
        """
        Unfreeze encoder parameters for fine-tuning.
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"All parameters unfrozen. Trainable parameters: {trainable_params:,}")
