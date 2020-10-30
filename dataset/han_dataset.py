from torch.utils.data import Dataset, DataLoader
import torch
from nltk import sent_tokenize, word_tokenize
from utils.utils import label2idx


class HANDataset(Dataset):

  def __init__(self, all_words, all_labels, word_vocab, is_train = True):

      super(HANDataset, self).__init__()
      
      self.all_words = all_words
      self.all_labels = all_labels
      self.is_train = is_train
      self.word_vocab = word_vocab
  
  def __len__(self):
      return len(self.all_labels)
  
  def collate_fn(self, batch):

      X,y = list(zip(*batch))
      max_n_sents = -1
      max_length_sent = -1

      for text in X:
          max_n_sents = max(max_n_sents, len(text))
          for words in text:
              max_length_sent = max(max_length_sent, len(words))

      X = pad_in_batch(X, max_n_sents, max_length_sent, self.word_vocab)
      X = torch.LongTensor(X)
      y = torch.LongTensor(y)

      word_masked, sent_masked = build_mask(X, self.word_vocab)

      # print(X)
      # print(y)

      return X, y, word_masked, sent_masked
  
  def __getitem__(self, idx):

      sents = self.all_words[idx]
      label = self.all_labels[idx]

      sents_idx = convert_token_to_idx(sents, self.word_vocab)

      # print(sents_idx)

      return sents_idx, label


def tokenize_sentence(text):
    return sent_tokenize(text)


def tokenize_word(text):
    return word_tokenize(text)

def build_raw_dataset(X, y, min_length = 3, sort_by_sents = True):
    
    all_words = []
    all_labels = []

    for text in X:
        sents = tokenize_sentence(text)
        sents_words = []
        for sent in sents:
            words = sent.split()
            if len(words) >= min_length:
                sents_words.append(words)

        all_words.append(sents_words)
    
    for label in y:
        label_idx = label2idx[label]
        all_labels.append(label_idx) 

    # if sort_by_sents:
    #     all_words = sorted(all_words, key=lambda x:len(x), reverse=True)
    
    assert len(all_words) == len(all_labels)
    return all_words, all_labels


def convert_token_to_idx(sents, word_vocab):

    sents_idx = []
    for sent in sents:
        
        words_idx = []
        for word in sent:
            if word in word_vocab:
                word_idx = word_vocab[word]
            else:
                word_idx = word_vocab['<UNK>']
            words_idx.append(word_idx)

        sents_idx.append(words_idx)
    return sents_idx


def build_mask(X, word_vocab):

    mask  = (X!= word_vocab['<PAD>']).type(torch.FloatTensor)

    sent_mask = []

    for j,m in enumerate(mask.clone()):
        sl = []
        for i,t in enumerate(m):
            if torch.sum(t) != 0:
                s_t = 1
            else:
                s_t = 0
            sl.append(s_t)
        sent_mask.append(sl)
    sent_mask = torch.FloatTensor(sent_mask)

    assert mask.shape == X.shape
    return mask, sent_mask


def pad_in_batch(X, max_sents, max_words, word_vocab):

    for sents in X:
        pad_sents_length = max_sents - len(sents)
        if pad_sents_length > 0:
            pad_sents = [[word_vocab['<PAD>']]] * pad_sents_length
            sents.extend(pad_sents)
    
    for sents in X:
        for words in sents:
            pad_length_words = max_words - len(words)
            pad_words = [word_vocab['<PAD>']] * pad_length_words
            words += pad_words
    
    return X