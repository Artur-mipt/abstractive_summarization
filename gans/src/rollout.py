import copy
import torch
import torch.nn.functional as F


class ROLLOUT:
    def __init__(self, gen, max_seq_len=30, vocab_size=10**4, gpu=True):
        self.gen = gen
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.gpu = gpu

    def rollout_mc_search(self, src, sentences, given_num):
        """
        fill up remain tokens with MC search
        :param sentences: size of batch_size * max_seq_len
        :param given_num: size of current state
        :return:
        """
        batch_size = sentences.size(0)
        self.max_seq_len = sentences.size(1)
        
        hidden, cell, encoder_outputs = self.gen.encoder(src)
        
        inp = sentences[:, :given_num]
        inp = inp[:, -1]
        out, hidden, cell = self.gen.decoder(inp, hidden, cell, encoder_outputs)
        out = out.view(batch_size, -1, self.vocab_size)[:, -1]

        samples = torch.zeros(batch_size, self.max_seq_len).long()
        samples[:, :given_num] = sentences[:, :given_num]

        # MC search
        self.max_seq_len = sentences.size(1)
        for i in range(given_num, self.max_seq_len):
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.view(-1).data
            inp = out.view(-1)
            out, hidden, cell = self.gen.decoder(inp, hidden, cell, encoder_outputs)

        return samples


    def get_reward(self, src, sentences, rollout_num, dis, device='cuda:0'):
        """
        get reward via Monte Carlo search
        :param sentences: size of batch_size * max_seq_len
        :param rollout_num:
        :param dis:
        :return: reward: [batch_size, seq_len]
        """
        self.max_seq_len = sentences.size(1)
        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards = torch.zeros([rollout_num * self.max_seq_len, batch_size]).float()
            idx = 0
            for i in range(rollout_num):
                for given_num in range(1, self.max_seq_len + 1):
                    samples = self.rollout_mc_search(src, sentences, given_num).to(device)
                    reward = dis.forward(samples).squeeze(1)
                    # out = F.softmax(out, dim=-1)
                    # reward = out[:, 1]
                    rewards[idx] = reward
                    idx += 1

        # rewards = torch.mean(rewards, dim=0)
        rewards = torch.mean(rewards.view(batch_size, self.max_seq_len, rollout_num), dim=-1)
        return rewards.to(device)