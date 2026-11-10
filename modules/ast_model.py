import torch
import torch.nn as nn
from transformers import ASTModel


class ASTFaultDetector(nn.Module):

    def __init__(self):
        super().__init__()

        self.ast = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

        # Freeze the pretrained AST initially
        for param in self.ast.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(
                self.ast.config.hidden_size,
                256
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, input_values):

        outputs = self.ast(
            input_values=input_values
        )

        # Mean pooling across time/patch tokens
        pooled = outputs.last_hidden_state.mean(
            dim=1
        )

        logits = self.classifier(pooled)

        return logits