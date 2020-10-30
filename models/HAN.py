import torch
import torch.nn as nn


class AttentionWithContext(nn.Module):

    def __init__(self, hidden_size):

        super(AttentionWithContext, self).__init__()

        self.w = nn.Linear(hidden_size, hidden_size)
        self.context = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, inputs, masked=None):

        #input = [batch, seq_len, hidden_size]

        u = torch.tanh(self.w(inputs))

        #u = [batch_seq_len, hidden_size]

        a = self.context(u)

        #a = [batch, seq_len, 1]

        if masked is not None:
            a = a.masked_fill_(masked == 0, -1e10)

        score = torch.softmax(a, dim=1)

        #score = [batch, seq_len, 1]

        s = (score * inputs).sum(dim=1)

        #score = [batch, hidden_size]

        return s


class WordAttnNet(nn.Module):

    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_size,
            pad_idx,
            drop_out=0.2):

        super(WordAttnNet, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True)
        self.attn = AttentionWithContext(hidden_size=2 * hidden_size)

        self.drop_out = nn.Dropout(drop_out)

        self.embedding_dim = embedding_dim

    def forward(self, inputs, h_n, masked=None):

        #inputs = [batch_size, seq_len]

        embedded = self.drop_out(self.embedding(inputs))

        h_t, h_n = self.gru(embedded, h_n)

        attn_vec = self.attn(h_t, masked=masked)

        return attn_vec.unsqueeze(1), h_n


class SentAttnNet(nn.Module):

    def __init__(self, hidden_size):

        super(SentAttnNet, self).__init__()

        self.gru = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True)
        self.attn = AttentionWithContext(hidden_size=2 * hidden_size)

    def forward(self, inputs, masked=None):

        h_t, h_n = self.gru(inputs)

        attn_vec = self.attn(h_t, masked=masked)

        return attn_vec


class HANATT(nn.Module):

    def __init__(self,
                 n_classes,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 pad_idx=0,
                 drop_out=0.2):

        super(HANATT, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.word_attn = WordAttnNet(vocab_size=vocab_size,
                                     embedding_dim=embedding_dim,
                                     hidden_size=hidden_size,
                                     pad_idx=pad_idx,
                                     drop_out=drop_out)

        self.sent_attn = SentAttnNet(hidden_size=hidden_size)

        self.linear_1 = nn.Linear(hidden_size * 2, hidden_size)

        self.linear_out = nn.Linear(hidden_size, n_classes)

    def forward(self, word_ids, word_masked, sent_masked):

        #word_ids = [batch_size, n_seq, seq_len]

        word_ids = word_ids.permute(1, 0, 2)

        word_masked = word_masked.permute(1, 0, 2)

        #[n_seq, batch_size, seq_len]

        word_h_n = nn.init.zeros_(
            torch.Tensor(
                2,
                word_ids.shape[1],
                self.hidden_size)).cuda()

        word_s_list = []

        for i, sent in enumerate(word_ids):

            word_s, word_h_n = self.word_attn(
                sent, word_h_n, word_masked[i].unsqueeze(-1))

            word_s_list.append(word_s)

        sent_s = torch.cat(word_s_list, 1)

        doc_s = self.sent_attn(sent_s, sent_masked.unsqueeze(-1))

        doc_s = torch.relu(self.linear_1(doc_s))

        logits = self.linear_out(doc_s)

        return logits
