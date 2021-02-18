import time
import pickle
import argparse
from tqdm import tqdm

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils.models import *
from utils.dataset import *
from utils.loss import *
from utils.logger import Logger
from utils.build_tag import *
from F_score_calculate import fscore, recall_k
from pycocoevalcap.eval import calculate_metrics

torch.backends.cudnn.benchmark = True
device_ids = [2]
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.cuda.set_device(2)
# device_ids = [0,1,2,3]

class DebuggerBase:
    def __init__(self, args):
        self.args = args
        self.min_val_loss = 10000000000
        self.min_val_tag_loss = 1000000
        self.min_val_stop_loss = 1000000
        self.min_val_word_loss = 10000000

        self.min_train_loss = 10000000000
        self.min_train_tag_loss = 1000000
        self.min_train_stop_loss = 1000000
        self.min_train_word_loss = 10000000

        self.best_fscore = 0
        self.best_Bleus = 0

        self.params = None

        self._init_model_path()
        self.model_dir = self._init_model_dir()
        self.writer = self._init_writer()
        self.train_transform = self._init_train_transform()
        self.val_transform = self._init_val_transform()
        self.test_transform = self._init_test_transform()
        self.vocab = self._init_vocab()
        self.tagger = self._init_tagger()
        self.model_state_dict = self._load_mode_state_dict()

        self.train_data_loader = self._init_data_loader(self.args.train_file_list, self.train_transform)
        self.val_data_loader = self._init_data_loader(self.args.val_file_list, self.val_transform)
        self.test_data_loader = self._init_data_loader(self.args.test_file_list, self.test_transform)

        self.extractor = self._init_visual_extractor()
        self.mlc = self._init_mlc()
        self.co_attention = self._init_co_attention()
        self.sentence_model = self._init_sentence_model()
        self.word_model = self._init_word_model()

        self.ce_criterion = self._init_ce_criterion()
        self.mse_criterion = self._init_mse_criterion()

        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.logger = self._init_logger()
        self.writer.write("{}\n".format(self.args))

    def train(self):
        for epoch_id in range(self.start_epoch, self.args.epochs):
            time_start = time.time()
            train_tag_loss, train_stop_loss, train_word_loss, train_loss = self._epoch_train()
            val_tag_loss, val_stop_loss, val_word_loss, val_loss = self._epoch_val()
            torch.cuda.empty_cache()
            print('epoch:',epoch_id)
            print('train_loss:',train_loss)
            print('train_tag_loss:', train_tag_loss)
            print('train_stop_loss:', train_stop_loss)
            print('train_word_loss:', train_word_loss)
            Fscores, Bleus = self._epoch_test()

            if self.args.mode == 'train':
                self.scheduler.step(train_loss)
            else:
                self.scheduler.step(val_loss)
            self.writer.write(
                "[{} - Epoch {}] train loss:{} - val_loss:{} - lr:{}\n".format(self._get_now(),
                                                                               epoch_id,
                                                                               train_loss,
                                                                               val_loss,
                                                                               self.optimizer.param_groups[0]['lr']))
            self._save_model(epoch_id,
                             val_loss,
                             val_tag_loss,
                             val_stop_loss,
                             val_word_loss,
                             train_loss,
                             Fscores,
                             Bleus)
            self._log(train_tags_loss=train_tag_loss,
                      train_stop_loss=train_stop_loss,
                      train_word_loss=train_word_loss,
                      train_loss=train_loss,
                      val_tags_loss=val_tag_loss,
                      val_stop_loss=val_stop_loss,
                      val_word_loss=val_word_loss,
                      val_loss=val_loss,
                      lr=self.optimizer.param_groups[0]['lr'],
                      epoch=epoch_id)
            time_end = time.time()
            print('training_time_per_epoch', time_end - time_start)

    def _epoch_train(self):
        raise NotImplementedError

    def _epoch_val(self):
        raise NotImplementedError

    def _epoch_test(self):
        raise NotImplementedError

    def _init_train_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.RandomCrop(self.args.crop_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_val_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.crop_size, self.args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_test_transform(self):
        transform = transforms.Compose([
            # transforms.Resize((self.args.resize, self.args.resize)),
            transforms.Resize(self.args.resize),
            transforms.CenterCrop(self.args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_model_dir(self):
        model_dir = os.path.join(self.args.model_path, self.args.saved_model_name)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # model_dir = os.path.join(model_dir, '2')
        # model_dir = os.path.join(model_dir, self._get_now())
        model_dir = os.path.join(model_dir, self.args.visual_model_name)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        return model_dir

    def _init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        self.writer.write("Vocab Size:{}\n".format(len(vocab)))

        return vocab

    def _init_tagger(self):
        raise NotImplementedError

    def _load_mode_state_dict(self):
        self.start_epoch = 0
        try:
            model_state = torch.load(self.args.load_model_path)
            self.start_epoch = model_state['epoch']
            self.writer.write("[Load Model-{} Succeed!]\n".format(self.args.load_model_path))
            self.writer.write("Load From Epoch {}\n".format(model_state['epoch']))
            return model_state
        except Exception as err:
            self.writer.write("[Load Model Failed] {}\n".format(err))
            return None

    def _init_visual_extractor(self):
        model = VisualFeatureExtractor(model_name=self.args.visual_model_name,
                                       pretrained=self.args.pretrained,
                                       embed_size=self.args.embed_size,
                                       hidden_size=self.args.hidden_size)

        try:
            model_state = torch.load(self.args.load_visual_model_path)
            model.load_state_dict(model_state['model'])
            print('visual')
            self.writer.write("[Load Visual Extractor Succeed!]\n")
        except Exception as err:
            self.writer.write("[Load Model Failed] {}\n".format(err))
            print('not load visual')

        if not self.args.visual_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = nn.DataParallel(model, device_ids=device_ids)
            model = model.cuda()

        return model

    def _init_mlc(self):
        model = MLC(classes=self.args.classes,
                    sementic_features_dim=self.args.sementic_features_dim,
                    fc_in_features=self.extractor.module.out_features,
                    k=self.args.k)

        try:
            model_state = torch.load(self.args.load_mlc_model_path)
            model.load_state_dict(model_state['model'])
            self.writer.write("[Load MLC Succeed!]\n")
        except Exception as err:
            self.writer.write("[Load MLC Failed {}!]\n".format(err))

        if not self.args.mlc_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = nn.DataParallel(model, device_ids=device_ids)
            model = model.cuda()
        return model

    def _init_co_attention(self):
        model = CoAttention(version=self.args.attention_version,
                            embed_size=self.args.embed_size,
                            hidden_size=self.args.hidden_size,
                            visual_size=self.extractor.module.out_features,
                            k=self.args.k,
                            momentum=self.args.momentum)

        try:
            model_state = torch.load(self.args.load_co_model_path)
            model.load_state_dict(model_state['model'])
            self.writer.write("[Load Co-attention Succeed!]\n")
        except Exception as err:
            self.writer.write("[Load Co-attention Failed {}!]\n".format(err))

        if not self.args.co_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = nn.DataParallel(model, device_ids=device_ids)
            model = model.cuda()
        return model

    def _init_sentence_model(self):
        raise NotImplementedError

    def _init_word_model(self):
        raise NotImplementedError

    def _init_data_loader(self, file_list, transform):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 caption_json=self.args.caption_json,
                                 file_list=file_list,
                                 vocabulary=self.vocab,
                                 transform=transform,
                                 batch_size=self.args.batch_size,
                                 s_max=self.args.s_max,
                                 n_max=self.args.n_max,
                                 shuffle=True)
        return data_loader

    @staticmethod
    def _init_ce_criterion():
        return nn.CrossEntropyLoss(size_average=False, reduce=False)

    @staticmethod
    def _init_mse_criterion():
        # return nn.MSELoss()
        return nn.MultiLabelSoftMarginLoss(size_average=False, reduce=False)

    def _init_optimizer(self):
        return torch.optim.Adam(params=self.params, lr=self.args.learning_rate)

    def _log(self,
             train_tags_loss,
             train_stop_loss,
             train_word_loss,
             train_loss,
             val_tags_loss,
             val_stop_loss,
             val_word_loss,
             val_loss,
             lr,
             epoch):
        info = {
            'train tags loss': train_tags_loss,
            'train stop loss': train_stop_loss,
            'train word loss': train_word_loss,
            'train loss': train_loss,
            'val tags loss': val_tags_loss,
            'val stop loss': val_stop_loss,
            'val word loss': val_word_loss,
            'val loss': val_loss,
            'learning rate': lr
        }

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, epoch + 1)

    def _init_logger(self):
        logger = Logger(os.path.join(self.model_dir, 'logs'))
        return logger

    def _init_writer(self):
        writer = open(os.path.join(self.model_dir, 'logs.txt'), 'w')
        return writer

    def _to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def _get_date(self):
        return str(time.strftime('%Y%m%d', time.gmtime()))

    def _get_now(self):
        return str(time.strftime('%Y%m%d-%H:%M', time.gmtime())).replace(':', '')

    def _init_scheduler(self):
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.args.patience, factor=0.1)
        return scheduler

    def _init_model_path(self):
        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)

    def _init_log_path(self):
        if not os.path.exists(self.args.log_path):
            os.makedirs(self.args.log_path)

    def _save_model(self,
                    epoch_id,
                    val_loss,
                    val_tag_loss,
                    val_stop_loss,
                    val_word_loss,
                    train_loss,
                    Fscores,
                    Bleus):
        def save_whole_model(_filename):
            self.writer.write("Saved Model in {}\n".format(_filename))
            torch.save({'extractor': self.extractor.state_dict(),
                        'mlc': self.mlc.state_dict(),
                        'co_attention': self.co_attention.state_dict(),
                        'sentence_model': self.sentence_model.state_dict(),
                        'word_model': self.word_model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch_id},
                       os.path.join(self.model_dir, "{}".format(_filename)))

        def save_part_model(_filename, value):
            self.writer.write("Saved Model in {}\n".format(_filename))
            torch.save({"model": value},
                       os.path.join(self.model_dir, "{}".format(_filename)))

        if val_loss < self.min_val_loss:
            file_name = "val_best_loss.pth.tar"
            save_whole_model(file_name)
            self.min_val_loss = val_loss

        if train_loss < self.min_train_loss:
            file_name = "train_best_loss.pth.tar"
            save_whole_model(file_name)
            self.min_train_loss = train_loss


class LSTMDebugger(DebuggerBase):
    def _init_(self, args):
        DebuggerBase.__init__(self, args)
        self.args = args

    def _epoch_train(self):
        tag_loss, stop_loss, word_loss, loss = 0, 0, 0, 0
        self.extractor.train()
        self.mlc.train()
        self.co_attention.train()
        self.sentence_model.train()
        self.word_model.train()
        batch_count = 0
        for i, (images, image_id, label, captions, prob) in enumerate(self.train_data_loader):
            batch_count +=1
            print('batch_count:',batch_count)
            batch_tag_loss, batch_stop_loss, batch_word_loss, batch_loss = 0, 0, 0, 0
            images = self._to_var(images)
            self.optimizer.zero_grad()
            visual_features, avg_features, V, v_g = self.extractor.forward(images)
            tags, semantic_features = self.mlc.forward(avg_features)
            batch_tag_loss = self.mse_criterion(tags, self._to_var(label, requires_grad=False)).sum()

            sentence_states = None
            prev_hidden_states = self._to_var(torch.zeros(images.shape[0], 1, self.args.hidden_size))

            context = self._to_var(torch.Tensor(captions).long(), requires_grad=False)
            prob_real = self._to_var(torch.Tensor(prob).long(), requires_grad=False)

            pred_sentences = {}
            real_sentences = {}
            for i in image_id:
                pred_sentences[i] = {}
                real_sentences[i] = {}
            for sentence_index in range(captions.shape[1]):
                ctx, _, _ = self.co_attention.forward(avg_features,
                                                      semantic_features,
                                                      prev_hidden_states)

                topic, p_stop, hidden_states, sentence_states = self.sentence_model.forward(ctx,
                                                                                            prev_hidden_states,
                                                                                             sentence_states)

                p_sto = prob_real
                batch_stop_loss += self.ce_criterion(p_stop.squeeze(), prob_real[:, sentence_index]).sum()
                words, states, atten_weights, beta = self.word_model.forward(topic, context[:, sentence_index, :], V,
                                                                             v_g)
                real_word = words.max(2)[1]

                sampled_ids = np.zeros((np.shape(topic)[0], captions.shape[2]))
                for k in range(real_word.size()[1]):
                    predicted = real_word[:, k]
                    sampled_ids[:, k] = predicted.squeeze()
                sampled_ids = sampled_ids * p_sto[:, sentence_index].unsqueeze(1)

                for id, array in zip(image_id, sampled_ids):
                    pred_sentences[id][sentence_index] = self.__vec2sent(array.cpu().detach().numpy())

                for word_index in range(1, captions.shape[2]):
                    word_mask = (context[:, sentence_index, word_index] > 0).float()
                    batch_word_loss += (self.ce_criterion(words[:,word_index-1,:], context[:, sentence_index, word_index]) * word_mask).sum()

            for id, array in zip(image_id, captions):
                for i, sent in enumerate(array):
                    real_sentences[id][i] = self.__vec2sent(sent)
            print('pre:',str(pred_sentences))
            print('gt:',str(real_sentences))
            batch_loss = self.args.lambda_tag * batch_tag_loss \
                         + self.args.lambda_stop * batch_stop_loss \
                         + self.args.lambda_word * batch_word_loss

            batch_loss.backward()
            if self.args.clip > 0:
                torch.nn.utils.clip_grad_norm(self.sentence_model.parameters(), self.args.clip)
                torch.nn.utils.clip_grad_norm(self.word_model.parameters(), self.args.clip)
            self.optimizer.step()
            torch.cuda.empty_cache()

            tag_loss += self.args.lambda_tag * batch_tag_loss.data
            stop_loss += self.args.lambda_stop * batch_stop_loss.data
            word_loss += self.args.lambda_word * batch_word_loss.data
            loss += batch_loss.data

        return tag_loss, stop_loss, word_loss, loss

    def _epoch_val(self):
        tag_loss, stop_loss, word_loss, loss = 0, 0, 0, 0
        self.extractor.eval()
        self.mlc.eval()
        self.co_attention.eval()
        self.sentence_model.eval()
        self.word_model.eval()

        for i, (images, image_id, label, captions, prob) in enumerate(self.val_data_loader):
            batch_tag_loss, batch_stop_loss, batch_word_loss, batch_loss = 0, 0, 0, 0
            images = self._to_var(images, requires_grad=False)

            visual_features, avg_features, V, v_g = self.extractor.forward(images)
            tags, semantic_features = self.mlc.forward(avg_features)

            batch_tag_loss = self.mse_criterion(tags, self._to_var(label, requires_grad=False)).sum()

            sentence_states = None
            prev_hidden_states = self._to_var(torch.zeros(images.shape[0], 1, self.args.hidden_size))

            context = self._to_var(torch.Tensor(captions).long(), requires_grad=False)
            prob_real = self._to_var(torch.Tensor(prob).long(), requires_grad=False)

            pred_sentences = {}
            real_sentences = {}
            for i in image_id:
                pred_sentences[i] = {}
                real_sentences[i] = {}

            for sentence_index in range(captions.shape[1]):
                ctx, v_att, a_att = self.co_attention.forward(avg_features,
                                                              semantic_features,
                                                              prev_hidden_states)

                topic, p_stop, hidden_states, sentence_states = self.sentence_model.forward(ctx,
                                                                                            prev_hidden_states,
                                                                                            sentence_states)
                p_sto = prob_real
                batch_stop_loss += self.ce_criterion(p_stop.squeeze(), prob_real[:, sentence_index]).sum()

                words, states, atten_weights, beta = self.word_model.forward(topic, context[:, sentence_index, :], V,
                                                                             v_g)

                real_word = words.max(2)[1]

                sampled_ids = np.zeros((np.shape(topic)[0], captions.shape[2]))
                for k in range(real_word.size()[1]):
                    predicted = real_word[:, k]
                    sampled_ids[:, k] = predicted.squeeze()
                sampled_ids = sampled_ids * p_sto[:, sentence_index].unsqueeze(1)

                for id, array in zip(image_id, sampled_ids):
                    pred_sentences[id][sentence_index] = self.__vec2sent(array.cpu().detach().numpy())


                for word_index in range(1, captions.shape[2]):
                    word_mask = (context[:, sentence_index, word_index] > 0).float()
                    batch_word_loss += (self.ce_criterion(words[:,word_index-1,:], context[:, sentence_index, word_index]) * word_mask).sum()

            for id, array in zip(image_id, captions):
                for i, sent in enumerate(array):
                    real_sentences[id][i] = self.__vec2sent(sent)
            print('-----------val----------')
            print('pre:',str(pred_sentences))
            print('gt:',str(real_sentences))

            batch_loss = self.args.lambda_tag * batch_tag_loss \
                         + self.args.lambda_stop * batch_stop_loss \
                         + self.args.lambda_word * batch_word_loss

            tag_loss += self.args.lambda_tag * batch_tag_loss.data
            stop_loss += self.args.lambda_stop * batch_stop_loss.data
            word_loss += self.args.lambda_word * batch_word_loss.data
            loss += batch_loss.data

        return tag_loss, stop_loss, word_loss, loss

    def _epoch_test(self):
        self.extractor.eval()
        self.mlc.eval()
        self.co_attention.eval()
        self.sentence_model.eval()
        self.word_model.eval()

        results = {}
        prediction = {}
        prediction_5 = {}
        prediction_20 = {}
        ground_truth = {}
        for i, (images, image_id, label, captions, prob) in enumerate(self.test_data_loader):
            torch.cuda.empty_cache()

            images = self._to_var(images, requires_grad=False)
            visual_features, avg_features, V, v_g = self.extractor.forward(images)
            tags, semantic_features = self.mlc.forward(avg_features)

            sentence_states = None
            prev_hidden_states = self._to_var(torch.zeros(images.shape[0], 1, self.args.hidden_size))
            pred_sentences = {}
            real_sentences = {}
            for i in image_id:
                pred_sentences[i] = {}
                real_sentences[i] = {}

            for i in range(self.args.s_max):
                ctx, alpha_v, alpha_a = self.co_attention.forward(avg_features, semantic_features, prev_hidden_states)

                topic, p_stop, hidden_state, sentence_states = self.sentence_model.forward(ctx,
                                                                                          prev_hidden_states,
                                                                                          sentence_states)
                p_stop = p_stop.squeeze(1)
                p_stop = torch.max(p_stop, 1)[1].unsqueeze(1)

                # Build the starting token Variable <start> (index 1): B x 1
                captions_w = Variable(torch.LongTensor(images.size(0), 1).fill_(self.vocab('<start>')).cuda())
                # Get generated caption idx list, attention weights and sentinel score
                # sampled_ids = []
                sampled_ids = np.zeros((np.shape(topic)[0], self.args.n_max))
                sampled_ids[:, 0] = captions_w.view(-1, )
                attention = []
                Beta = []
                # Initial hidden states
                states = None
                for j in range(1, self.args.n_max):
                    scores, states, atten_weights, beta = self.word_model.forward(topic, captions_w, V, v_g, states)
                    predicted = scores.max(2)[1]  # argmax
                    captions_w = predicted
                    sampled_ids[:, j] = predicted.squeeze()
                    attention.append(atten_weights)
                    Beta.append(beta)

                attention = torch.cat(attention, dim=1)
                Beta = torch.cat(Beta, dim=1)
                prev_hidden_states = hidden_state
                sampled_ids = sampled_ids * p_stop

                for id, array in zip(image_id, sampled_ids):
                    pred_sentences[id][i] = self.__vec2sent(array.cpu().detach().numpy())

            for id, array in zip(image_id, captions):
                for i, sent in enumerate(array):
                    real_sentences[id][i] = self.__vec2sent(sent)

            for id, pred_tag, real_tag in zip(image_id, tags, label):
                results[id] = {
                    'Real Tags': self.tagger.inv_tags2array(real_tag),
                    'Pred Tags': self.tagger.array2tags(torch.topk(pred_tag, self.args.k)[1].cpu().detach().numpy()),
                    'Pred Sent': pred_sentences[id],
                    'Real Sent': real_sentences[id]
                }
            for id, pred_tag, real_tag in zip(image_id, tags, label):
                real_tag = self.tagger.inv_tags2array(real_tag)
                real_tag = ';'.join(real_tag)
                real_tag = real_tag.strip(';')
                pred_tag_5 = self.tagger.array2tags(torch.topk(pred_tag, 5)[1].cpu().detach().numpy())
                pred_tag_5 = ';'.join(pred_tag_5)
                pred_tag_5 = pred_tag_5.strip(';')
                pred_tag_20 = self.tagger.array2tags(torch.topk(pred_tag, 20)[1].cpu().detach().numpy())
                pred_tag_20 = ';'.join(pred_tag_20)
                pred_tag_20 = pred_tag_20.strip(';')
                pred_tag_10 = self.tagger.array2tags(torch.topk(pred_tag, self.args.k)[1].cpu().detach().numpy())
                pred_tag_10 = ';'.join(pred_tag_10)
                pred_tag_10 = pred_tag_10.strip(';')
                ground_truth[id] = real_tag
                prediction[id] = pred_tag_10
                prediction_5[id] = pred_tag_5
                prediction_20[id] = pred_tag_20
        prf_scores = fscore(prediction, ground_truth, self.args.visual_model_name)
        recall_5 = recall_k(prediction_5, ground_truth, self.args.visual_model_name, k=5)
        recall_20 = recall_k(prediction_20, ground_truth, self.args.visual_model_name, k=20)
        Bleu = self._evaluation_captions(results)
        self.__save_json(results)
        return prf_scores, Bleu

    def _evaluation_captions(self, result):
        test = result
        datasetGTS = {'annotations': []}
        datasetRES = {'annotations': []}

        for i, image_id in enumerate(test):
            array = []
            for each in test[image_id]['Pred Sent']:
                array.append(test[image_id]['Pred Sent'][each])
            pred_sent = '. '.join(array)

            array = []
            for each in test[image_id]['Real Sent']:
                sent = test[image_id]['Real Sent'][each]
                if len(sent) != 0:
                    array.append(sent)
            real_sent = '. '.join(array)
            datasetGTS['annotations'].append({
                'image_id': i,
                'caption': real_sent
            })
            datasetRES['annotations'].append({
                'image_id': i,
                'caption': pred_sent
            })

        rng = range(len(test))
        eva_scores = calculate_metrics(rng, datasetGTS, datasetRES)
        print(eva_scores)
        test_result_dir = self.args.result_path + self.args.visual_model_name + '/' + 'BLEUS.txt'
        with open(test_result_dir, 'a') as f:
            f.writelines(str(eva_scores))
            f.writelines('\n')
        Bleu = eva_scores['Bleu_1']
        return Bleu

    def __save_json(self, result):
        result_path = os.path.join(self.args.result_path, self.args.visual_model_name)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path, '{}.json'.format(self.args.result_name)), 'w') as f:
            json.dump(result, f)

    def _init_tagger(self):
        return Tag()

    def __vec2sent(self, array):
        sampled_caption = []
        for word_id in array:
            word = self.vocab.get_word_by_id(word_id)
            if word == '<start>'or word == '<pad>':
                continue
            if word == '<end>':
                break
            sampled_caption.append(word)
        return ' '.join(sampled_caption)

    def _init_sentence_model(self):
        model = SentenceLSTM(version=self.args.sent_version,
                             embed_size=self.args.embed_size,
                             hidden_size=self.args.hidden_size,
                             num_layers=self.args.sentence_num_layers,
                             dropout=self.args.dropout,
                             momentum=self.args.momentum)

        try:
            model_state = torch.load(self.args.load_sentence_model_path)
            model.load_state_dict(model_state['model'])
            self.writer.write("[Load Sentence Model Succeed!\n")
        except Exception as err:
            self.writer.write("[Load Sentence model Failed {}!]\n".format(err))

        if not self.args.sentence_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = nn.DataParallel(model, device_ids=device_ids)
            model = model.cuda()
        return model

    def _init_word_model(self):
        model = WordLSTM(vocab_size=len(self.vocab),
                         embed_size=self.args.embed_size,
                         hidden_size=self.args.hidden_size,
                         num_layers=self.args.word_num_layers,
                         n_max=self.args.n_max)

        try:
            model_state = torch.load(self.args.load_word_model_path)
            model.load_state_dict(model_state['model'])
            self.writer.write("[Load Word Model Succeed!\n")
        except Exception as err:
            self.writer.write("[Load Word model Failed {}!]\n".format(err))

        if not self.args.word_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = nn.DataParallel(model, device_ids=device_ids)
            model = model.cuda()
        return model


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    """
    Data Argument
    """
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--mode', type=str, default='train')

    # Path Argument
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='the path for vocabulary object')
    parser.add_argument('--image_dir', type=str, default='./data/images',
                        help='the path for images')
    parser.add_argument('--caption_json', type=str, default='./data/captions.json',
                        help='path for captions')
    parser.add_argument('--train_file_list', type=str, default='./data/train_data.txt',
                        help='the train array')
    parser.add_argument('--val_file_list', type=str, default='./data/val_data.txt',
                        help='the val array')
    parser.add_argument('--test_file_list', type=str, default='./data/test_data.txt',
                        help='the test array')
    # transforms argument
    parser.add_argument('--resize', type=int, default=256,
                        help='size for resizing images')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    # Load/Save model argument
    parser.add_argument('--model_path', type=str, default='./report_models/',
                        help='path for saving trained models')
    parser.add_argument('--load_model_path', type=str, default='',
                        help='The path of loaded model')
    parser.add_argument('--saved_model_name', type=str, default='model',
                        help='The name of saved model')

    """
    Model Argument
    """
    parser.add_argument('--momentum', type=int, default=0.1)
    # VisualFeatureExtractor
    parser.add_argument('--visual_model_name', type=str, default='resnet152',
                        help='CNN model name')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='not using pretrained model when training')
    parser.add_argument('--load_visual_model_path', type=str,
                        default='.')
    parser.add_argument('--visual_trained', action='store_true', default=True,
                        help='Whether train visual extractor or not')

    # MLC
    parser.add_argument('--classes', type=int, default=587)
    parser.add_argument('--sementic_features_dim', type=int, default=512)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--load_mlc_model_path', type=str,
                        default='.')
    parser.add_argument('--mlc_trained', action='store_true', default=True)

    # Co-Attention
    parser.add_argument('--attention_version', type=str, default='v2')
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--load_co_model_path', type=str, default='.')
    parser.add_argument('--co_trained', action='store_true', default=True)

    # Sentence Model
    parser.add_argument('--sent_version', type=str, default='v2')
    parser.add_argument('--sentence_num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--load_sentence_model_path', type=str,
                        default='.')
    parser.add_argument('--sentence_trained', action='store_true', default=True)

    # Word Model
    parser.add_argument('--word_num_layers', type=int, default=1)
    parser.add_argument('--load_word_model_path', type=str,
                        default='.')
    parser.add_argument('--word_trained', action='store_true', default=True)

    """
    Training Argument
    """
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=int, default=0.0003)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--clip', type=float, default=0.35,
                        help='gradient clip, -1 means no clip (default: 0.35)')
    parser.add_argument('--s_max', type=int, default=6)
    parser.add_argument('--n_max', type=int, default=30)

    # Loss Function
    parser.add_argument('--lambda_tag', type=float, default=1)
    parser.add_argument('--lambda_stop', type=float, default=1)
    parser.add_argument('--lambda_word', type=float, default=1)

    # Saved result
    parser.add_argument('--result_path', type=str, default='./results/model/',
                        help='the path for storing results')
    parser.add_argument('--result_name', type=str, default='test_result',
                        help='the name of results')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    debugger = LSTMDebugger(args)
    debugger.train()
