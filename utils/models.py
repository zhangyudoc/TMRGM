import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence


class VisualFeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet152', pretrained=False, embed_size=512, hidden_size=512):
        super(VisualFeatureExtractor, self).__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, self.out_features, self.avg_func = self.__get_model()
        self.avgpool = nn.AvgPool2d(7)
        self.affine_a = nn.Linear(2048, hidden_size)  # v_i = W_a * A
        self.affine_b = nn.Linear(2048, embed_size)  # v_g = W_b * a^g

        # Dropout before affine transformation
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ReLU()

    def __get_model(self):
        model = None
        out_features = None
        func = None
        if self.model_name == 'vgg19':
            vgg = models.vgg19(pretrained=self.pretrained)
            modules = list(vgg.features)
            model = nn.Sequential(*modules)
            func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            out_features = 512
        elif self.model_name == 'resnet152':
            resnet = models.resnet152(pretrained=self.pretrained)
            modules = list(resnet.children())[:-2]
            model = nn.Sequential(*modules)
            out_features = resnet.fc.in_features
            func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        return model, out_features, func


    def forward(self, images):
        visual_features = self.model(images)
        avg_features = self.avg_func(visual_features).squeeze()
        a_g = self.avgpool(visual_features)
        a_g = a_g.view(a_g.size(0), -1)
        V = visual_features.view(visual_features.size(0), visual_features.size(1), -1).transpose(1, 2)
        V = F.relu(self.affine_a(self.dropout(V)))
        v_g = F.relu(self.affine_b(self.dropout(a_g)))
        return visual_features, avg_features, V, v_g


class MLC(nn.Module):
    def __init__(self,
                 classes=587,
                 sementic_features_dim=512,
                 fc_in_features=2048,
                 k=10):
        super(MLC, self).__init__()
        self.classifier = nn.Linear(in_features=fc_in_features, out_features=classes)
        self.embed = nn.Embedding(classes, sementic_features_dim)
        self.k = k
        self.softmax = nn.Softmax()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, avg_features):
        tags = self.softmax(self.classifier(avg_features))
        semantic_features = self.embed(torch.topk(tags, self.k)[1])
        return tags, semantic_features


class CoAttention(nn.Module):
    def __init__(self,
                 version='v4',
                 embed_size=512,
                 hidden_size=512,
                 visual_size=512,
                 k=10,
                 momentum=0.1):
        super(CoAttention, self).__init__()
        self.version = version
        self.W_v = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_h = nn.Linear(in_features=hidden_size, out_features=visual_size)
        self.bn_v_h = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_att = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v_att = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_a = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a = nn.BatchNorm1d(num_features=k, momentum=momentum)

        self.W_a_h = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_h = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_a_att = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_att = nn.BatchNorm1d(num_features=k, momentum=momentum)

        self.W_fc = nn.Linear(in_features=visual_size + hidden_size, out_features=embed_size)
        self.bn_fc = nn.BatchNorm1d(num_features=embed_size, momentum=momentum)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.__init_weight()

    def __init_weight(self):
        self.W_v.weight.data.uniform_(-0.1, 0.1)
        self.W_v.bias.data.fill_(0)

        self.W_v_h.weight.data.uniform_(-0.1, 0.1)
        self.W_v_h.bias.data.fill_(0)

        self.W_v_att.weight.data.uniform_(-0.1, 0.1)
        self.W_v_att.bias.data.fill_(0)

        self.W_a.weight.data.uniform_(-0.1, 0.1)
        self.W_a.bias.data.fill_(0)

        self.W_a_h.weight.data.uniform_(-0.1, 0.1)
        self.W_a_h.bias.data.fill_(0)

        self.W_a_att.weight.data.uniform_(-0.1, 0.1)
        self.W_a_att.bias.data.fill_(0)

        self.W_fc.weight.data.uniform_(-0.1, 0.1)
        self.W_fc.bias.data.fill_(0)

    def forward(self, avg_features, semantic_features, h_sent):
        if self.version == 'v1':
            return self.v1(avg_features, semantic_features, h_sent)
        elif self.version == 'v2':
            return self.v2(avg_features, semantic_features, h_sent)

    def v1(self, avg_features, semantic_features, h_sent) -> object:
        W_v = self.bn_v(self.W_v(avg_features))
        W_v_h = self.bn_v_h(self.W_v_h(h_sent.squeeze(1)))

        alpha_v = self.softmax(self.bn_v_att(self.W_v_att(self.tanh(W_v + W_v_h))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.bn_a_h(self.W_a_h(h_sent))
        W_a = self.bn_a(self.W_a(semantic_features))
        alpha_a = self.softmax(self.bn_a_att(self.W_a_att(self.tanh(torch.add(W_a_h, W_a)))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a


    def v2(self, avg_features, semantic_features, h_sent):
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(torch.add(W_v, W_v_h))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a


class SentenceLSTM(nn.Module):
    def __init__(self,
                 version='v3',
                 embed_size=512,
                 hidden_size=512,
                 num_layers=1,
                 dropout=0.3,
                 momentum=0.1):
        super(SentenceLSTM, self).__init__()
        self.version = version

        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout)

        self.W_t_h = nn.Linear(in_features=hidden_size,
                               out_features=embed_size,
                               bias=True)
        self.bn_t_h = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_t_ctx = nn.Linear(in_features=embed_size,
                                 out_features=embed_size,
                                 bias=True)
        self.bn_t_ctx = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_stop_s_1 = nn.Linear(in_features=hidden_size,
                                    out_features=embed_size,
                                    bias=True)
        self.bn_stop_s_1 = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_stop_s = nn.Linear(in_features=hidden_size,
                                  out_features=embed_size,
                                  bias=True)
        self.bn_stop_s = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_stop = nn.Linear(in_features=embed_size,
                                out_features=2,
                                bias=True)
        self.bn_stop = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_topic = nn.Linear(in_features=embed_size,
                                 out_features=embed_size,
                                 bias=True)
        self.bn_topic = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.__init_weight()

    def __init_weight(self):
        self.W_t_h.weight.data.uniform_(-0.1, 0.1)
        self.W_t_h.bias.data.fill_(0)

        self.W_t_ctx.weight.data.uniform_(-0.1, 0.1)
        self.W_t_ctx.bias.data.fill_(0)

        self.W_stop_s_1.weight.data.uniform_(-0.1, 0.1)
        self.W_stop_s_1.bias.data.fill_(0)

        self.W_stop_s.weight.data.uniform_(-0.1, 0.1)
        self.W_stop_s.bias.data.fill_(0)

        self.W_stop.weight.data.uniform_(-0.1, 0.1)
        self.W_stop.bias.data.fill_(0)

        self.W_topic.weight.data.uniform_(-0.1, 0.1)
        self.W_topic.bias.data.fill_(0)

    def forward(self, ctx, prev_hidden_state, states=None) -> object:
        if self.version == 'v1':
            return self.v1(ctx, prev_hidden_state, states)
        elif self.version == 'v2':
            return self.v2(ctx, prev_hidden_state, states)

    def v1(self, ctx, prev_hidden_state, states=None):
        ctx = ctx.unsqueeze(1)
        hidden_state, states = self.lstm(ctx, states)
        topic = self.W_topic(self.sigmoid(self.bn_t_h(self.W_t_h(hidden_state))
                                          + self.bn_t_ctx(self.W_t_ctx(ctx))))
        p_stop = self.W_stop(self.sigmoid(self.bn_stop_s_1(self.W_stop_s_1(prev_hidden_state))
                                          + self.bn_stop_s(self.W_stop_s(hidden_state))))
        return topic, p_stop, hidden_state, states


    def v2(self, ctx, prev_hidden_state, states=None):
        ctx = ctx.unsqueeze(1)
        hidden_state, states = self.lstm(ctx, states)
        topic = self.W_topic(self.tanh(self.W_t_h(hidden_state) + self.W_t_ctx(ctx)))
        p_stop = self.W_stop(self.tanh(self.W_stop_s_1(prev_hidden_state) + self.W_stop_s(hidden_state)))
        return topic, p_stop, hidden_state, states


class Atten(nn.Module):
    def __init__(self, hidden_size):
        super(Atten, self).__init__()

        self.affine_v = nn.Linear(hidden_size, 49, bias=False)  # W_v
        self.affine_g = nn.Linear(hidden_size, 49, bias=False)  # W_g
        self.affine_s = nn.Linear(hidden_size, 49, bias=False)  # W_s
        self.affine_h = nn.Linear(49, 1, bias=False)  # w_h

        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform(self.affine_v.weight)
        init.xavier_uniform(self.affine_g.weight)
        init.xavier_uniform(self.affine_h.weight)
        init.xavier_uniform(self.affine_s.weight)

    def forward(self, V, h_t, s_t):
        content_v = self.affine_v(self.dropout(V)).unsqueeze(1) \
                    + self.affine_g(self.dropout(h_t)).unsqueeze(2)

        z_t = self.affine_h(self.dropout(F.tanh(content_v))).squeeze(3)
        alpha_t = F.softmax(z_t.view(-1, z_t.size(2))).view(z_t.size(0), z_t.size(1), -1)

        c_t = torch.bmm(alpha_t, V).squeeze(2)

        content_s = self.affine_s(self.dropout(s_t)) + self.affine_g(self.dropout(h_t))
        z_t_extended = self.affine_h(self.dropout(F.tanh(content_s)))

        extended = torch.cat((z_t, z_t_extended), dim=2)
        alpha_hat_t = F.softmax(extended.view(-1, extended.size(2))).view(extended.size(0), extended.size(1), -1)
        beta_t = alpha_hat_t[:, :, -1]

        beta_t = beta_t.unsqueeze(2)
        c_hat_t = beta_t * s_t + (1 - beta_t) * c_t

        return c_hat_t, alpha_t, beta_t


class Sentinel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Sentinel, self).__init__()

        self.affine_x = nn.Linear(input_size, hidden_size, bias=False)
        self.affine_h = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(0.5)

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform(self.affine_x.weight)
        init.xavier_uniform(self.affine_h.weight)

    def forward(self, x_t, h_t_1, cell_t):
        gate_t = self.affine_x(self.dropout(x_t)) + self.affine_h(self.dropout(h_t_1))
        gate_t = F.sigmoid(gate_t)

        s_t = gate_t * F.tanh(cell_t)

        return s_t


class AdaptiveBlock(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size):
        super(AdaptiveBlock, self).__init__()

        self.sentinel = Sentinel(embed_size * 3, hidden_size)

        self.atten = Atten(hidden_size)

        self.mlp = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(0.5)

        self.hidden_size = hidden_size
        self.init_weights()

    def init_weights(self):
        init.kaiming_normal(self.mlp.weight, mode='fan_in')
        self.mlp.bias.data.fill_(0)

    def forward(self, x, hiddens, cells, V):

        h0 = self.init_hidden(x.size(0))[0].transpose(0, 1)

        if hiddens.size(1) > 1:
            hiddens_t_1 = torch.cat((h0, hiddens[:, :-1, :]), dim=1)
        else:
            hiddens_t_1 = h0

        sentinel = self.sentinel(x, hiddens_t_1, cells)

        c_hat, atten_weights, beta = self.atten(V, hiddens, sentinel)

        scores = self.mlp(self.dropout(c_hat + hiddens))

        return scores, atten_weights, beta

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        if torch.cuda.is_available():
            return (Variable(weight.new(1, bsz, self.hidden_size).zero_().cuda()),
                    Variable(weight.new(1, bsz, self.hidden_size).zero_().cuda()))
        else:
            return (Variable(weight.new(1, bsz, self.hidden_size).zero_()),
                    Variable(weight.new(1, bsz, self.hidden_size).zero_()))


class WordLSTM(nn.Module):
    def __init__(self,
                 embed_size,
                 hidden_size,
                 vocab_size,
                 num_layers,
                 n_max=50):
        super(WordLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size*3, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.adaptive = AdaptiveBlock(embed_size, hidden_size, vocab_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.__init_weights()
        self.n_max = n_max
        self.vocab_size = vocab_size

    def __init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, topic_vec, captions, V, v_g, states=None):
        embeddings = self.embed(captions)
        v_g_expand = v_g.unsqueeze(1).expand_as(embeddings)
        topic_vec_expand = topic_vec.expand_as(embeddings)
        x = torch.cat((embeddings, topic_vec_expand), dim=2)
        x = torch.cat((x, v_g_expand), dim=2)

        if torch.cuda.is_available():
            hiddens = Variable(torch.zeros(x.size(0), x.size(1), self.hidden_size).cuda())
            cells = Variable(torch.zeros(x.size(1), x.size(0), self.hidden_size).cuda())
        else:
            hiddens = Variable(torch.zeros(x.size(0), x.size(1), self.hidden_size))
            cells = Variable(torch.zeros(x.size(1), x.size(0), self.hidden_size))

        for time_step in range(x.size(1)):
            x_t = x[:, time_step, :]
            x_t = x_t.unsqueeze(1)

            h_t, states = self.lstm(x_t, states)

            hiddens[:, time_step, :] = h_t[:, -1, :]  # Batch_first
            cells[time_step, :, :] = states[1]

        cells = cells.transpose(0, 1)

        scores, atten_weights, beta = self.adaptive(x, hiddens, cells, V)

        return scores, states, atten_weights, beta

    def sample(self, features, start_tokens):
        sampled_ids = np.zeros((np.shape(features.cpu())[0], self.n_max))##增加.cpu
        sampled_ids[:, 0] = start_tokens.cpu().view(-1, )
        predicted = start_tokens
        embeddings = features
        embeddings = embeddings

        for i in range(1, self.n_max):
            predicted = self.embed(predicted)
            embeddings = torch.cat([embeddings, predicted], dim=1)
            hidden_states, _ = self.lstm(embeddings)
            hidden_states = hidden_states[:, -1, :]
            outputs = self.linear(hidden_states)
            predicted = torch.max(outputs, 1)[1]
            sampled_ids[:, i] = predicted
            predicted = predicted.unsqueeze(1)
        return sampled_ids


    