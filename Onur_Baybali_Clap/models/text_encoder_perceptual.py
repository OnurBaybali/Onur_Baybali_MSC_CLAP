# text_encoder_perceptual.py

import torch.nn as nn
from transformers import (
    BertModel,
    BertTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    RobertaModel,
    RobertaTokenizer,
    DistilBertModel,
    DistilBertTokenizer,
    CLIPTokenizer,
    CLIPTextModel,
)

MODELS = {
    'openai/clip-vit-base-patch32': (CLIPTextModel, CLIPTokenizer, 512),
    'prajjwal1/bert-tiny': (BertModel, BertTokenizer, 128),
    'prajjwal1/bert-mini': (BertModel, BertTokenizer, 256),
    'prajjwal1/bert-small': (BertModel, BertTokenizer, 512),
    'prajjwal1/bert-medium': (BertModel, BertTokenizer, 512),
    'gpt2': (GPT2Model, GPT2Tokenizer, 768),
    'distilgpt2': (GPT2Model, GPT2Tokenizer, 768),
    'bert-base-uncased': (BertModel, BertTokenizer, 768),
    'bert-large-uncased': (BertModel, BertTokenizer, 1024),
    'roberta-base': (RobertaModel, RobertaTokenizer, 768),
    'roberta-large': (RobertaModel, RobertaTokenizer, 1024),
    'distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer, 768),
    'distilroberta-base': (RobertaModel, RobertaTokenizer, 768),
}


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_type = config["text_encoder_args"]["type"]
        self.text_model = MODELS[model_type][0].from_pretrained(model_type, add_pooling_layer=False)

        if config["text_encoder_args"]["freeze"]:
            for param in self.text_model.parameters():
                param.requires_grad = False

        self.pad_token_id = MODELS[model_type][1].from_pretrained(model_type).pad_token_id
        self.text_width = MODELS[model_type][-1]

    def forward(self, input_ids):
        # attention mask hesapla
        attention_mask = (input_ids != self.pad_token_id).long().to(input_ids.device)

        # modelden geçir
        output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state