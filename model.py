import torch
import torch.nn as nn

from module import *


def get_rep_mask(lengths, device):
	batch_size = len(lengths)
	seq_len = torch.max(lengths)
	rep_mask = torch.FloatTensor(batch_size, seq_len).to(torch.device(device))
	rep_mask.data.fill_(1)
	for i in range(batch_size):
		rep_mask[i, lengths[i]:] = 0

	return rep_mask.unsqueeze_(-1)


class NN4SNLI(nn.Module):

	def __init__(self, args, data):
		super(NN4SNLI, self).__init__()

		self.class_size = args.class_size
		self.dropout = args.dropout
		self.d_e = args.d_e
		self.d_ff = args.d_ff
		self.device = args.device

		self.word_emb = nn.Embedding(args.word_vocab_size, args.word_dim)
		# initialize word embedding with GloVe
		self.word_emb.weight.data.copy_(data.TEXT.vocab.vectors)
		# fine-tune the word embedding
		self.word_emb.weight.requires_grad = False
		# <unk> vectors is randomly initialized
		nn.init.uniform_(self.word_emb.weight.data[0], -0.05, 0.05)

		self.sentence_encoder = SentenceEncoder(args)

		self.fc = nn.Linear(args.d_e * 4 * 4, args.d_e)
		self.fc_out = nn.Linear(args.d_e, args.class_size)

		self.layer_norm = nn.LayerNorm(args.d_e)
		self.dropout = nn.Dropout(args.dropout)
		self.relu = nn.ReLU()

	def forward(self, batch):
		premise, pre_lengths = batch.premise
		hypothesis, hypo_lengths = batch.hypothesis

		# (batch, seq_len, word_dim)
		pre_x = self.word_emb(premise)
		hypo_x = self.word_emb(hypothesis)

		# (batch, seq_len, 1)
		pre_rep_mask = get_rep_mask(pre_lengths, self.device)
		hypo_rep_mask = get_rep_mask(hypo_lengths, self.device)

		# (batch, seq_len, 4 * d_e)
		pre_s = self.sentence_encoder(pre_x, pre_rep_mask)
		hypo_s = self.sentence_encoder(hypo_x, hypo_rep_mask)

		# (batch, seq_len, 4 * 4 * d_e)
		s = torch.cat([pre_s, hypo_s, (pre_s - hypo_s).abs(), pre_s * hypo_s], dim=-1)

		s = self.dropout(s)
		outputs = self.relu(self.layer_norm(self.fc(s)))
		outputs = self.dropout(outputs)
		outputs = self.fc_out(outputs)

		return outputs

