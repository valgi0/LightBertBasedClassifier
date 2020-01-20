from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import BertTokenizer
import math
import numpy as np
import torch
from torch.autograd import Variable
import json
import pickle as p
from tqdm.autonotebook import tqdm, trange
from pathlib import *


class Config():

    class ConfigIn(): 
  
        def __init__(self):
          self.set_default()

        def set_default(self):
          self.bert_max_tokens = 256
          self.bert_model_dim = 512
          self.bert_layers = 4
          self.bert_multi_head = 1
          self.classifier_layer = 5
          self.classifier_hidden_size = 1024
          self.reduced_dimension = 32
          self.numb_of_classes = 1
          self.config_directory = 'configLB.json'
          self.debug_mode = True
    
    instance = None

    def __init__(self, path='./', default = False):
        path = Path(path)
        if path.exists and path.is_dir and not default:
            if path.joinpath('Config.json').exists:
                Config.instance = self.load_file(path)
            else:
                raise NameError('Config.json not found in path. Use default to create a new one')
        elif default:
            Config.instance = self.ConfigIn()
            if path.exists and path.is_dir:
                self.save_file(path.joinpath('Config.json'))
                print('File Config.json saved at {}'.format(str(path)))
            else:
                raise NameError('Path not found')


    def save_file(self, path = './'):
      with open(path, 'w') as fp:
        json.dump(self.instance, fp, default = lambda o: o.__dict__,
                  sort_keys=True, indent=4)
 
    def load_file(self, path):
        with path.open('r') as fp:
            c = json.load(fp, indent=4)
        return c

class LightBertBasedClassifier(torch.nn.Module):
  '''
  This class represent a classifier that use transformer encoder for generate
  embeddings, a reducer layer to reduce the embeddings dimension and a classifier
  to classify the relations between the two input string
  '''
  def __init__(self, config):
    super(LightBertBasedClassifier, self).__init__()
    self.embedder = self.create_embeddings(config)
    self.bert_layer1 = self.create_bert_layer(config)
    self.bert_layer2 = self.create_bert_layer(config)
    self.feature_reducer_layer1 = self.create_feature_reducer_layer(config)
    self.feature_reducer_layer2 = self.create_feature_reducer_layer(config)
    self.classifier = self.create_classifier(config)
    self.config = config
    self.pe = self.create_positional_encoder(config)
  
  def create_embeddings(self, config):
    layer = torch.nn.Sequential(
        torch.nn.Linear(1, config.bert_model_dim),
        torch.nn.Tanh(),
        torch.nn.Dropout(),
        torch.nn.Linear(config.bert_model_dim, config.bert_model_dim),
        torch.nn.Tanh(),
        torch.nn.Dropout(),
        torch.nn.Linear(config.bert_model_dim, config.bert_model_dim),
        torch.nn.Tanh(),
        )
    return layer
  
  def create_bert_layer(self, config):
    '''
    This layer is similar to bert indeed both of them are encoder from transformers
    This layer takes at least 64 tokens with 512 features
    It supports multi head attentions but it is disabled by default configuration
    This layer contains more encoder sub layers: default 4
    Each sublayer has two pieces:
    - MultiHead self attention
    - Feed Forward Network
    TransformerEncoderLayer is the class used to create this layer
    '''
    # First we create a single sublayer
    tlayer = TransformerEncoderLayer(
        config.bert_model_dim,
        config.bert_multi_head,
        dim_feedforward=config.bert_model_dim * 2
    )
    
    # and then we create the entire layer telling to it the layer end how many
    bert_layer = TransformerEncoder(tlayer, config.bert_layers)
    return bert_layer
  
  def create_feature_reducer_layer(self, config):
    '''
    This layer is used to reduce the dimensionality of the embeddings. By default the are
    reduced from 512 to 32
    This layer is composed by a Feed Forward Network with 2 layers
    '''
    layer = torch.nn.Sequential(
        torch.nn.Linear(config.bert_model_dim, config.reduced_dimension),
        torch.nn.Tanh(),
        torch.nn.Linear(config.reduced_dimension, config.reduced_dimension),
        torch.nn.Tanh(),
        )
    return layer
  
  def create_classifier(self, config):
    '''
    This is the classification Layer
    It is a Feed Forward Network with 3 layers and an output size bigger than
    the number of classes
    '''
    layer = torch.nn.Sequential(
        torch.nn.Linear(config.reduced_dimension * 2, config.classifier_hidden_size),
        torch.nn.Tanh(),
        torch.nn.Dropout(),
        torch.nn.Linear(config.classifier_hidden_size, config.classifier_hidden_size),
        torch.nn.Tanh(),
        torch.nn.Dropout(),
        torch.nn.Linear(config.classifier_hidden_size, config.numb_of_classes),
        torch.nn.Tanh(),
        )
    return layer
  
  def create_positional_encoder(self, config):
    pe = PositionalEncoder(
        config.bert_model_dim, max_seq_len=config.bert_max_tokens)
    return pe

  def forward(self, sentences1, sentences2, mask1=None, mask2=None, debug_mode = False):
    '''
    Args:
      - sentences1 are the ids of the sentence 1 in the shape of (batch_size, max_tokens)
      - sentences2 are the ids of the sentence 2 in the same shape as sentences1
      - mask1 and mask2 are the masks of sentences1 and 2. They are 1 for attention mechanism
        value is 0 or 1 and 0 means ignore.
      - debug_mode is true to print some debug informations 
    '''
    debug = debug_mode
    debug_string = '[DEBUG] -> {}'
    if debug:
      print(debug_string.format('Sentences1: ' + str(sentences1.shape) + ', Sentence2: ' + str(sentences2.shape)))
    embeddings = torch.stack([self.embedder(x) for x in sentences1])
    embeddings2 = torch.stack([self.embedder(x) for x in sentences2])
    embeddings = self.pe(embeddings)
    embeddings2 = self.pe(embeddings2)
    if debug:
      print(debug_string.format('Inputs1 shape: ' + str(np.shape(embeddings)) + ' Inputs2 shape: ' + str(np.shape(embeddings2))))
    batch = len(embeddings)
    out1 = self.bert_layer1(embeddings, mask1)
    out2 = self.bert_layer2(embeddings2, mask2)
    if debug:
      print(debug_string.format('BERT done'))
      print(debug_string.format('Out1 shape: ' + str(out1.shape) + 'Out2 shape: ' + str(out2.shape)))
    
    out1 = torch.mean(out1, dim=1)
    out2 = torch.mean(out2, dim=1)
    if debug:
      print(debug_string.format('Mean between tokens embeddings done'))
      print(debug_string.format('Out1 shape: ' + str(out1.shape) + 'Out2 shape: ' + str(out2.shape)))
      print(debug_string.format('Out1: ' + str(out1) + 'Out2: ' + str(out2)))
    rout1 = self.feature_reducer_layer1(out1)
    rout2 = self.feature_reducer_layer2(out2)
    if debug:
      print(debug_string.format('Dimensions reduced to ' + str(self.config.reduced_dimension)))
      print(debug_string.format('rout1: ' + str(rout1) + 'rout2: ' + str(rout2)))
    out = torch.cat((rout1, rout2), dim=1)
    if debug:
      print(debug_string.format('Two outputs concatenations'))
      print(debug_string.format(str(out)))
    final = self.classifier(out)
    return final


