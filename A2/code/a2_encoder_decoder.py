# Copyright 2020 University of Toronto, all rights reserved

'''Concrete implementations of abstract base classes.

You don't need anything more than what's been imported here
'''

import torch

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase


class Encoder(EncoderBase):

    def init_submodules(self):
        # initialize parameterized submodules here: rnn, embedding
        # using: self.source_vocab_size, self.word_embedding_size, self.pad_id,
        # self.dropout, self.cell_type, self.hidden_state_size,
        # self.num_hidden_layers
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        # relevant pytorch modules:
        # torch.nn.{LSTM, GRU, RNN, Embedding}
        self.embedding = torch.nn.Embedding(self.source_vocab_size, self.word_embedding_size, padding_idx=self.pad_id)
        if self.cell_type == "rnn":
            self.rnn = torch.nn.RNN(self.word_embedding_size, self.hidden_state_size, self.num_hidden_layers,
                                    bidirectional=True, dropout=self.dropout)
        elif self.cell_type == "gru":
            self.rnn = torch.nn.GRU(self.word_embedding_size, self.hidden_state_size, self.num_hidden_layers,
                                    bidirectional=True, dropout=self.dropout)
        elif self.cell_type == "lstm":
            self.rnn = torch.nn.LSTM(self.word_embedding_size, self.hidden_state_size, self.num_hidden_layers,
                                     bidirectional=True, dropout=self.dropout)

    def get_all_rnn_inputs(self, F):
        # compute input vectors for each source transcription.
        # F is shape (S, N)
        # x (output) is shape (S, N, I)
        mask = (F != self.pad_id).float().unsqueeze(-1)  # shape [S,N,1]
        x = self.embedding(F)  # shape [S,N,I]
        x *= mask  # 无效的输入数据，置为0
        return x

    def get_all_hidden_states(self, x, F_lens, h_pad):
        # compute all final hidden states for provided input sequence.
        # make sure you handle padding properly!
        # x is of shape (S, N, I)
        # F_lens is of shape (N,)
        # h_pad is a float
        # h (output) is of shape (S, N, 2 * H)
        # relevant pytorch modules:
        # torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence
        mask = (x == 0).all(dim=-1)  # shape = [S,N]
        x = torch.nn.utils.rnn.pack_padded_sequence(x, F_lens, enforce_sorted=False)
        h, _ = self.rnn(x)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h)  # shape [S, N, 2 * H]
        h[mask] = h_pad
        return h


class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # initialize parameterized submodules: embedding, cell, ff
        # using: self.target_vocab_size, self.word_embedding_size, self.pad_id,
        # self.hidden_state_size, self.cell_type
        # relevant pytorch modules:
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        # torch.nn.{Embedding,Linear,LSTMCell,RNNCell,GRUCell}
        self.embedding = torch.nn.Embedding(self.target_vocab_size, self.word_embedding_size, padding_idx=self.pad_id)
        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(input_size=self.word_embedding_size, hidden_size=self.hidden_state_size)
        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(input_size=self.word_embedding_size, hidden_size=self.hidden_state_size)
        elif self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(input_size=self.word_embedding_size, hidden_size=self.hidden_state_size)
        self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)

    def get_first_hidden_state(self, h, F_lens):
        # build decoder's first hidden state. Ensure it is derived from encoder
        # hidden state that has processed the entire sequence in each
        # direction:
        # - Populate indices 0 to self.hidden_state_size // 2 - 1 (inclusive)
        #   with the hidden states of the encoder's forward direction at the
        #   highest index in time *before padding*
        # - Populate indices self.hidden_state_size // 2 to
        #   self.hidden_state_size - 1 (inclusive) with the hidden states of
        #   the encoder's backward direction at time t=0
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # htilde_tm1 (output) is of shape (N, 2 * H)
        # relevant pytorch modules: torch.cat
        S, N, hidden_state_size = h.size()
        last_idx = (F_lens - 1).long()  # shape = (N,)
        last_idx = last_idx.view(1, N, 1).expand(1, N, hidden_state_size)  # shape = [1, N, 2*H]
        htilde_tm0 = h.gather(dim=0, index=last_idx).squeeze(0)  # [N, 2 * H]
        return htilde_tm0

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # determine the input to the rnn for *just* the current time step.
        # No attention.
        # E_tm1 is of shape (N,)
        # htilde_tm1 is of shape (N, 2 * H) or a tuple of two of those (LSTM)
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # xtilde_t (output) is of shape (N, Itilde)
        mask = (E_tm1 != self.pad_id).float().unsqueeze(1)  # (N,1)
        xtilde_t = self.embedding(E_tm1)
        xtilde_t *= mask
        return xtilde_t

    def get_current_hidden_state(self, xtilde_t, htilde_tm1):
        # update the previous hidden state to the current hidden state.
        # xtilde_t is of shape (N, Itilde)
        # htilde_tm1 is of shape (N, 2 * H) or a tuple of two of those (LSTM)
        # htilde_t (output) is of same shape as htilde_tm1
        htilde_t = self.cell(xtilde_t, htilde_tm1)
        return htilde_t

    def get_current_logits(self, htilde_t):
        # determine un-normalized log-probability distribution over output
        # tokens for current time step.
        # htilde_t is of shape (N, 2 * H), even for LSTM (cell state discarded)
        # logits_t (output) is of shape (N, V)
        logits_t = self.ff(htilde_t)
        return logits_t


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # same as before, but with a slight modification for attention
        # using: self.target_vocab_size, self.word_embedding_size, self.pad_id,
        # self.hidden_state_size, self.cell_type
        # cell_type will be one of: ['lstm', 'gru', 'rnn']
        self.embedding = torch.nn.Embedding(self.target_vocab_size, self.word_embedding_size, padding_idx=self.pad_id)
        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(input_size=self.word_embedding_size + self.hidden_state_size,
                                          hidden_size=self.hidden_state_size)
        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(input_size=self.word_embedding_size + self.hidden_state_size,
                                         hidden_size=self.hidden_state_size)
        elif self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(input_size=self.word_embedding_size + self.hidden_state_size,
                                         hidden_size=self.hidden_state_size)
        self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)
        self.additive_attention_layer = torch.nn.Linear(self.hidden_state_size * 2, 1)

    def get_first_hidden_state(self, h, F_lens):
        # same as before, but initialize to zeros
        # relevant pytorch modules: torch.zeros_like
        # ensure result is on same device as h!
        htilde_0 = torch.zeros_like(h[0]).to(h.device)  # （N，2*H）
        return htilde_0

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # update to account for attention. Use attend() for c_t
        mask = (E_tm1 != self.pad_id).float().unsqueeze(1)  # (N,1)
        T_E_tm1 = self.embedding(E_tm1)
        T_E_tm1 *= mask  # (N, Itilde)
        c_t = self.attend(htilde_tm1[0] if self.cell_type == 'lstm' else htilde_tm1, h, F_lens)  # (N, 2*H)
        xtilde_t = torch.cat([T_E_tm1, c_t], dim=1)  # (N, Itilde+ 2 * H)
        return xtilde_t

    def attend(self, htilde_t, h, F_lens):
        # compute context vector c_t. Use get_attention_weights() to calculate
        # alpha_t.
        # htilde_t is of shape (N, 2 * H)
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # c_t (output) is of shape (N, 2 * H)
        alpha_t = self.get_attention_weights(htilde_t, h, F_lens)  # (S, N)
        alpha_t = alpha_t.unsqueeze(2).expand(-1, -1, self.hidden_state_size)  # (S,N,2*H)

        return (alpha_t * h).sum(dim=0)  # (N, 2*H)

    def get_attention_weights(self, htilde_t, h, F_lens):
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_energy_scores()
        # alpha_t (output) is of shape (S, N)
        e_t = self.get_energy_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= F_lens  # (S, N)
        e_t = e_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(e_t, 0)

    def get_energy_scores(self, htilde_t, h):
        # Determine energy scores via cosine similarity
        # htilde_t is of shape (N, 2 * H)
        # h is of shape (S, N, 2 * H)
        # e_t (output) is of shape (S, N)
        scores_type = "scaled-dot-product"  # {cosine, additive , dot-product, scaled-dot-product}
        eps = 1e-8
        S = h.size()[0]
        htilde_t = htilde_t.unsqueeze(0).expand(S, -1, -1)  # (S, N ,2 * H)
        if scores_type == "cosine":
            htilde_t_norm = torch.norm(htilde_t, dim=2).unsqueeze(-1)  # (S,N,1)
            h_norm = torch.norm(h, dim=2).unsqueeze(-1)  # (S,N,1)
            htilde_t = htilde_t / torch.max(htilde_t_norm, eps * torch.ones_like(htilde_t_norm))
            h = h / torch.max(h_norm, eps * torch.ones_like(h_norm))
            e_t = (htilde_t * h).sum(dim=2)  # (S, N)
        elif scores_type == "additive":
            e_t = torch.cat([htilde_t, h], dim=2)  # (S, N, 4*H)
            e_t = self.additive_attention_layer(e_t).squeeze(2)  # (S, N)
            e_t = torch.nn.functional.tanh(e_t)
        elif scores_type == "dot-product":
            e_t = (htilde_t * h).sum(dim=2)
        elif scores_type == "scaled-dot-product":
            htilde_t_norm = torch.norm(htilde_t, dim=2)  # (S,N)
            htilde_t_norm = torch.sqrt(htilde_t_norm)
            e_t = (htilde_t * h).sum(dim=2) / torch.max(htilde_t_norm, eps * torch.ones_like(htilde_t_norm))
        else:
            raise NotImplementedError
        return e_t


class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(self, encoder_class, decoder_class):
        # initialize the parameterized submodules: encoder, decoder
        # encoder_class and decoder_class inherit from EncoderBase and
        # DecoderBase, respectively.
        # using: self.source_vocab_size, self.source_pad_id,
        # self.word_embedding_size, self.encoder_num_hidden_layers,
        # self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        # self.target_vocab_size, self.target_eos
        # Recall that self.target_eos doubles as the decoder pad id since we
        # never need an embedding for it
        self.encoder = encoder_class(source_vocab_size=self.source_vocab_size, pad_id=self.source_pad_id,
                                     word_embedding_size=self.word_embedding_size,
                                     num_hidden_layers=self.encoder_num_hidden_layers,
                                     hidden_state_size=self.encoder_hidden_size, dropout=self.encoder_dropout,
                                     cell_type=self.cell_type)
        self.decoder = decoder_class(target_vocab_size=self.target_vocab_size, pad_id=self.target_eos,
                                     word_embedding_size=self.word_embedding_size,
                                     hidden_state_size=self.encoder_hidden_size * 2, cell_type=self.cell_type)

    def get_logits_for_teacher_forcing(self, h, F_lens, E):
        # get logits over entire E. logits predict the *next* word in the
        # sequence.
        # h is of shape (S, N, 2 * H)
        # F_lens is of shape (N,)
        # E is of shape (T, N)
        # logits (output) is of shape (T - 1, N, Vo)
        # relevant pytorch modules: torch.{zero_like,stack}
        # hint: recall an LSTM's cell state is always initialized to zero.
        # Note logits sequence dimension is one shorter than E (why?)
        T, N = E.size()
        htilde_tm1 = self.decoder.get_first_hidden_state(h, F_lens)
        if self.cell_type == 'lstm':
            cell_state = torch.zeros_like(htilde_tm1).to(htilde_tm1.device)
        logits = []
        for i in range(T - 1):
            E_tm1 = E[i, :]
            xtilde_t = self.decoder.get_current_rnn_input(E_tm1, htilde_tm1, h, F_lens)
            if self.cell_type == 'lstm':
                htilde_tm1, cell_state = self.decoder.get_current_hidden_state(xtilde_t, (htilde_tm1, cell_state))
            else:
                htilde_tm1 = self.decoder.get_current_hidden_state(xtilde_t, htilde_tm1)
            logits.append(self.decoder.get_current_logits(htilde_tm1))
        logits = torch.stack(logits, dim=0)
        return logits

    def update_beam(self, htilde_t, b_tm1_1, logpb_tm1, logpy_t):
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        # htilde_t is of shape (N, K, 2 * H) or a tuple of two of those (LSTM)
        # logpb_tm1 is of shape (N, K)
        # b_tm1_1 is of shape (t, N, K)
        # b_t_0 (first output) is of shape (N, K, 2 * H) or a tuple of two of
        #                                                         those (LSTM)
        # b_t_1 (second output) is of shape (t + 1, N, K)
        # logpb_t (third output) is of shape (N, K)
        # relevant pytorch modules:
        # torch.{flatten,topk,unsqueeze,expand_as,gather,cat}
        # hint: if you flatten a two-dimensional array of shape z of (A, B),
        # then the element z[a, b] maps to z'[a*B + b]
        N, K, V = logpy_t.size()
        logpb_tm1 = logpb_tm1.unsqueeze(-1).expand(-1, -1, V)
        logpb_t = logpb_tm1 + logpy_t  # (N,K,V)
        logpb_t, indices = logpb_t.view(N, -1).topk(self.beam_width, dim=1)  # (N,K), (N,K)
        indices_k = indices // V  # (N, K)
        indices_v = indices % V  # (N, K)
        if self.cell_type == 'lstm':
            b_t_0 = (htilde_t[0].gather(dim=1, index=indices_k.unsqueeze(-1).expand_as(htilde_t[0]))
                     , htilde_t[1].gather(dim=1, index=indices_k.unsqueeze(-1).expand_as(htilde_t[1])))
        else:
            b_t_0 = htilde_t.gather(dim=1, index=indices_k.unsqueeze(-1).expand_as(htilde_t))  # (N,K,2*H)
        b_tm1_1 = b_tm1_1.gather(dim=2, index=indices_k.unsqueeze(0).expand_as(b_tm1_1))  # (t,N,K)
        b_t_1 = torch.cat([b_tm1_1, indices_v.unsqueeze(0)], dim=0)
        return b_t_0, b_t_1, logpb_t
