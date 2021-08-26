import torch
from transformers import RobertaModel

phobert_model = RobertaModel.from_pretrained('vinai/phobert-base')
input_names = ['input_ids', 'attention_mask']
output_names = ['last_hidden_state', 'pooler_output']
dynamic_axes = {
    'input_ids': {0: 'batch', 1: 'length'},
    'attention_mask': {0: 'batch', 1: 'length'},
    'last_hidden_state': {0: 'batch', 1: 'length', 2: 'hidden_size'},
    'pooler_output': {0: 'batch', 1: 'hidden_size'}
}

pad_idx = 1

dummy_input_ids = torch.ones(1, 100).long()
dummy_mask = torch.ones(1, 100).long()

torch.onnx.export(phobert_model, (dummy_input_ids, dummy_mask), 'phobert_base.onnx',
                  verbose=True, input_names=input_names, output_names=output_names,
                  dynamic_axes=dynamic_axes, opset_version=12)
