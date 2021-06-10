from __future__ import absolute_import
from __future__ import unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from .highway import Highway


# using [char-lstm-last_hidden_state, word_embedding] as actual embedding, like bilistm-crf
class LstmTokenEmbedder(nn.Module):
  def __init__(self, config, word_emb_layer, char_emb_layer, use_cuda=False):
    '''
    ARGS:
    : config: model configuration, corresponds to ELMo models options \ref page 2 footnote
    : word_emb_layer: word embedding layer, defined in embedding_layer.py
    : char_emb_layer: char embedding layer, defined in embedding_layer.py
    : used_cude: whether to use cuda
    '''
    super(LstmTokenEmbedder, self).__init__()
    self.config = config
    self.use_cuda = use_cuda
    self.word_emb_layer = word_emb_layer
    self.char_emb_layer = char_emb_layer
    self.output_dim = config['encoder']['projection_dim']   # word embedding dim
    emb_dim = 0   # acutal embedding dimesion
    if word_emb_layer is not None:
      emb_dim += word_emb_layer.n_d   # n_d, embedding dimension

    if char_emb_layer is not None:
      emb_dim += char_emb_layer.n_d * 2
      self.char_lstm = nn.LSTM(char_emb_layer.n_d, char_emb_layer.n_d, num_layers=1, bidirectional=True,
                               batch_first=True, dropout=config['dropout'])
    
    # combine char-embedding and word-embedding, then project
    self.projection = nn.Linear(emb_dim, self.output_dim, bias=True)

  def forward(self, word_inp, chars_inp, shape):
    embs = []
    batch_size, seq_len = shape
    if self.word_emb_layer is not None:
      word_emb = self.word_emb_layer(Variable(word_inp).cuda() if self.use_cuda else Variable(word_inp))
      embs.append(word_emb)

    if self.char_emb_layer is not None:
      chars_inp = chars_inp.view(batch_size * seq_len, -1)  # flat <batch_size, seq_len, word_len> into <batch_size*seq_len, word_len>
      chars_emb = self.char_emb_layer(Variable(chars_inp).cuda() if self.use_cuda else Variable(chars_inp))  #<batch_size*seq_len, word_len, d_char>
      _, (chars_outputs, __) = self.char_lstm(chars_emb)
      chars_outputs = chars_outputs.contiguous().view(-1, self.config['token_embedder']['char_dim'] * 2)  # bidirecional lstm, so last dim*2
      embs.append(chars_outputs)

    token_embedding = torch.cat(embs, dim=2)

    return self.projection(token_embedding)


class ConvTokenEmbedder(nn.Module):
  def __init__(self, config, word_emb_layer, char_emb_layer, use_cuda):
    super(ConvTokenEmbedder, self).__init__()
    '''
    ARGS:
    : config: model configuration, corresponds to ELMo models options \ref page 2 footnote
    : word_emb_layer: word embedding layer, defined in embedding_layer.py
    : char_emb_layer: char embedding layer, defined in embedding_layer.py
    : used_cude: whether to use cuda
    '''
    self.config = config
    self.use_cuda = use_cuda

    self.word_emb_layer = word_emb_layer
    self.char_emb_layer = char_emb_layer

    self.output_dim = config['encoder']['projection_dim']
    self.emb_dim = 0
    if word_emb_layer is not None:
      self.emb_dim += word_emb_layer.n_d

    if char_emb_layer is not None:
      self.convolutions = []
      cnn_config = config['token_embedder']
      filters = cnn_config['filters'] # list of (size, num_filters)
      char_embed_dim = cnn_config['char_dim']

      for i, (width, num) in enumerate(filters):
        conv = torch.nn.Conv1d(
          in_channels=char_embed_dim,
          out_channels=num,
          kernel_size=width,
          bias=True
        )
        self.convolutions.append(conv)

      self.convolutions = nn.ModuleList(self.convolutions)
      
      self.n_filters = sum(f[1] for f in filters)
      self.n_highway = cnn_config['n_highway']

      self.highways = Highway(self.n_filters, self.n_highway, activation=torch.nn.functional.relu)
      self.emb_dim += self.n_filters

    self.projection = nn.Linear(self.emb_dim, self.output_dim, bias=True)
    
  def forward(self, word_inp, chars_inp, shape):
    embs = []
    batch_size, seq_len = shape
    if self.word_emb_layer is not None:
      batch_size, seq_len = word_inp.size(0), word_inp.size(1)
      word_emb = self.word_emb_layer(Variable(word_inp).cuda() if self.use_cuda else Variable(word_inp))
      embs.append(word_emb)

    if self.char_emb_layer is not None:
      chars_inp = chars_inp.view(batch_size * seq_len, -1)  # flat into <batch_size*seq_len, word_len>

      character_embedding = self.char_emb_layer(Variable(chars_inp).cuda() if self.use_cuda else Variable(chars_inp)) # loop up <batch_size*seq_len, word_len, char_dim>

      character_embedding = torch.transpose(character_embedding, 1, 2)

      cnn_config = self.config['token_embedder']
      if cnn_config['activation'] == 'tanh':
        activation = torch.nn.functional.tanh
      elif cnn_config['activation'] == 'relu':
        activation = torch.nn.functional.relu
      else:
        raise Exception("Unknown activation")

      convs = []
      for i in range(len(self.convolutions)):
        convolved = self.convolutions[i](character_embedding)
        # (batch_size * sequence_length, n_filters for this width)
        convolved, _ = torch.max(convolved, dim=-1)   # <batch_size*seq_len, num_filter>
        convolved = activation(convolved)
        convs.append(convolved)           # list of <batch_size*seq_len, num_filter>
      char_emb = torch.cat(convs, dim=-1) # <batch_size*seq_len, num_filters>
      char_emb = self.highways(char_emb)  # <batch_size*seq_len, num_filters>

      embs.append(char_emb.view(batch_size, -1, self.n_filters))  # <batch_size, seq_len, num_filters>
      
    token_embedding = torch.cat(embs, dim=2)  # <batch_size, seq_len, num_filters + word_dim>

    return self.projection(token_embedding) # project
