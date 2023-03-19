import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, load_multitask_data, load_multitask_test_data, load_extra_data

from evaluation import model_eval_sst, model_eval_multitask, test_model_multitask

from classifier import SentimentDataset

import csv

TQDM_DISABLE=False

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

# Define the L2 regularization hyperparameter
l2_lambda = 0.01
la_lambda = 0.01

# Helper functions
def adversarial_loss_sst(model, b_ids_sst, b_mask_sst, b_labels_sst, epsilon=0.2):
    # Generate adversarial examples using FGSM attack
    b_ids_sst = b_ids_sst.to(torch.float32)

    perturbations = torch.zeros_like(b_ids_sst).uniform_(-epsilon, epsilon)
    perturbations = perturbations.cuda()
    perturbed_ids_sst = b_ids_sst + perturbations
    perturbed_ids_sst = torch.clamp(perturbed_ids_sst, min=0)

    # Compute logits for original and perturbed inputs
    perturbed_logits_sst = model.predict_sentiment((perturbed_ids_sst.type(torch.LongTensor)).cuda(), b_mask_sst)
    
    # Compute adversarial loss using cross-entropy
    adv_loss = F.cross_entropy(perturbed_logits_sst, b_labels_sst.view(-1), reduction='sum') / args.batch_size

    # b_ids_sst = b_ids_sst.long()
    return adv_loss

def adversarial_loss_sentence(model, b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels, epsilon=0.2, which='PARA'):
    # Generate adversarial examples using FGSM attack
    b_ids_1 = b_ids_1.to(torch.float32)
    b_ids_2 = b_ids_2.to(torch.float32)

    perturbations_1 = torch.zeros_like(b_ids_1).uniform_(-epsilon, epsilon)
    perturbations_2 = torch.zeros_like(b_ids_2).uniform_(-epsilon, epsilon)
    perturbations_1 = perturbations_1.cuda()
    perturbations_2 = perturbations_2.cuda()
    perturbed_ids_1 = b_ids_1 + perturbations_1
    perturbed_ids_2 = b_ids_2 + perturbations_2

    perturbed_ids_1 = torch.clamp(perturbed_ids_1, min=0)
    perturbed_ids_2 = torch.clamp(perturbed_ids_2, min=0)


    # Compute logits for original and perturbed inputs
    if (which == 'PARA'):
        perturbed_logits = model.predict_paraphrase((perturbed_ids_1.type(torch.LongTensor)).cuda(), b_mask_1, 
                                                         (perturbed_ids_2.type(torch.LongTensor)).cuda(), b_mask_2)
    else:
        perturbed_logits = model.predict_similarity((perturbed_ids_1.type(torch.LongTensor)).cuda(), b_mask_1, 
                                                         (perturbed_ids_2.type(torch.LongTensor)).cuda(), b_mask_2)
    
    # Compute adversarial loss using cross-entropy
    adv_loss = F.binary_cross_entropy_with_logits(perturbed_logits, b_labels.view(-1).float(), reduction='sum') / args.batch_size
    return adv_loss

# Load the data: a list of (sentence, label)
def cfimdb_data(filename, flag='train'):
    num_labels = {}
    data = []
    if flag == 'test':
        with open(filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                data.append((sent,sent_id))
    else:
        with open(filename, 'r', encoding='utf-8') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                data.append((sent, label,sent_id))
        print(f"load {len(data)} data from {filename}")

    if flag == 'train':
        return data, len(num_labels)
    else:
        return data

class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:
    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bertSST = BertModel.from_pretrained('bert-base-uncased')

        self.PARA_SBERT_1 = BertModel.from_pretrained('bert-base-uncased')
        self.PARA_SBERT_2 = BertModel.from_pretrained('bert-base-uncased')

        self.STS_SBERT_1 = BertModel.from_pretrained('bert-base-uncased')
        self.STS_SBERT_2 = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bertSST.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        for param in self.PARA_SBERT_1.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        for param in self.PARA_SBERT_2.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True   

        for param in self.STS_SBERT_1.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        for param in self.STS_SBERT_2.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True   

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_fc1 = nn.Linear(BERT_HIDDEN_SIZE, 64)
        self.sentiment_fc2 = nn.Linear(64, 5)
        self.paraphrase_fc1 = nn.Linear(BERT_HIDDEN_SIZE*3, 256)
        self.paraphrase_fc2 = nn.Linear(256, 1)


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        output = self.bertSST(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
        output_para = self.bertPARA(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
        output_STS = self.bertSTS(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']

        return output, output_para, output_STS


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        pooled_output = self.bertSST(input_ids, attention_mask)['pooler_output']

        sentiment_output = self.sentiment_fc1(pooled_output)
        sentiment_output = nn.ReLU()(sentiment_output)
        sentiment_output = self.sentiment_fc2(sentiment_output)

        return sentiment_output

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        pooled_output_1 = self.PARA_SBERT_1(input_ids=input_ids_1, attention_mask=attention_mask_1)['pooler_output']
        pooled_output_2 = self.PARA_SBERT_2(input_ids=input_ids_2, attention_mask=attention_mask_2)['pooler_output']
        abs_diff = torch.abs(pooled_output_1 - pooled_output_2)

        pooled_outputs = torch.cat([pooled_output_1, pooled_output_2, abs_diff], dim=1) # shape [batch_size, 3 * seq_len, hidden_size]

        paraphrase_output = self.paraphrase_fc1(pooled_outputs)
        paraphrase_output = F.relu(paraphrase_output)
        paraphrase_output = self.dropout(paraphrase_output)
        paraphrase_output = self.paraphrase_fc2(paraphrase_output)
        
        return paraphrase_output.squeeze()


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
           WE WANT TO APPLY THE SIGMOID to scale from 0 to 1 then multiply by 5
        '''
        pooled_output_1 = self.STS_SBERT_1(input_ids=input_ids_1, attention_mask=attention_mask_1)['pooler_output']
        pooled_output_2 = self.STS_SBERT_2(input_ids=input_ids_2, attention_mask=attention_mask_2)['pooler_output']

        similarity_output = F.cosine_similarity(pooled_output_1, pooled_output_2, dim=1)
        similarity_output = self.dropout(similarity_output)

        similarity_output = torch.sigmoid(similarity_output) * 5
        return similarity_output.squeeze()

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train') #'SICK_train-para.csv', 'SICK_train-scaled.csv', split ='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train') # 'SICK_dev-para.csv', 'SICK_dev-scaled.csv'
    extra_train_data = load_extra_data('./data/SICK_train-scaled.csv', split ='train')
    # extra_dev_data = load_extra_data('./data/SICK_dev-scaled.csv', split ='train')
    # _, _, _, extra_dev_data = load_extra_data(args.sst_dev,args.para_dev,'./data/SICK_dev-scaled.csv', split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    #-----------
    #load paraphrase dataset
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_dev_data.collate_fn)
    
    #-----------
    #load sts dataset
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=sts_dev_data.collate_fn)
    
    #-----------
    extra_train_data = SentencePairDataset(extra_train_data, args)
    # extra_dev_data = SentencePairDataset(extra_dev_data, args)

    extra_train_dataloader = DataLoader(extra_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=extra_train_data.collate_fn)
    
    # x_data, x_labels = cfimdb_data('data/cfimdb_scaled.csv', 'train')
    # cfimdb_train_dataset = SentimentDataset(x_data, args)

    # cfimdb_train_dataloader = DataLoader(cfimdb_train_dataset, shuffle=True, batch_size=args.batch_size,
    #                                      collate_fn=cfimdb_train_dataset.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    #checkpoint = torch.load('finetune-100-1e-05-best-multitask.pt')
    #model.load_state_dict(checkpoint['model'])
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_sen_acc = 0
    best_para_acc = 0
    best_sts_corr = 0

    # Compute the L2 regularization term
    l2_reg = torch.tensor(0.).cuda()
    for param in model.parameters():
        l2_reg += torch.norm(param)

    # Counter for Early Stopping, Half Learning Rate
    counter = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss_sst = 0
        train_loss_para = 0
        train_loss_sts = 0
        train_loss_extra = 0
        num_batches_sst = 0
        num_batches_para = 0
        num_batches_sts = 0
        num_batches_extra = 0

        for batch_sst in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # Sentiment analysis task
            b_ids_sst, b_mask_sst, b_labels_sst = (batch_sst['token_ids'],
                                                   batch_sst['attention_mask'], batch_sst['labels'])

            b_ids_sst = b_ids_sst.to(device)
            b_mask_sst = b_mask_sst.to(device)
            b_labels_sst = b_labels_sst.to(device)

            logits_sst = model.predict_sentiment(b_ids_sst, b_mask_sst)
            
            loss_sst = F.cross_entropy(logits_sst, b_labels_sst.view(-1), reduction='sum') / args.batch_size
            adv_loss_sst = adversarial_loss_sst(model, b_ids_sst, b_mask_sst, b_labels_sst)
            loss_sst += la_lambda * adv_loss_sst + l2_lambda * l2_reg

            optimizer.zero_grad()
            loss_sst.backward(retain_graph=True)
            optimizer.step()

            train_loss_sst += loss_sst.item()
            num_batches_sst += 1

        for batch_para in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            # Paraphrase detection task
            b_ids_para_1, b_mask_para_1, b_ids_para_2, b_mask_para_2, b_labels_para = (batch_para['token_ids_1'], batch_para['attention_mask_1'], 
                                                                                       batch_para['token_ids_2'], batch_para['attention_mask_2'], batch_para['labels'])

            b_ids_para_1 = b_ids_para_1.to(device)
            b_ids_para_2 = b_ids_para_2.to(device)
            b_mask_para_1 = b_mask_para_1.to(device)
            b_mask_para_2 = b_mask_para_2.to(device)
            b_labels_para = b_labels_para.to(device)

            logits_para = model.predict_paraphrase(b_ids_para_1, b_mask_para_1, b_ids_para_2, b_mask_para_2)
            loss_para = F.binary_cross_entropy_with_logits(logits_para, b_labels_para.view(-1).float(), reduction='sum') / args.batch_size

            print(logits_para)

            adv_loss_para = adversarial_loss_sentence(model, b_ids_para_1, b_mask_para_1, b_ids_para_2, b_mask_para_2, b_labels_para)
            loss_para += la_lambda * adv_loss_para + l2_lambda * l2_reg

            optimizer.zero_grad()
            loss_para.backward(retain_graph=True)
            optimizer.step()

            train_loss_para += loss_para.item()
            num_batches_para += 1

        # Consider separating training tasks
        
        for batch_sts in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):    
            # Textual Similarity Task
            b_ids_sts_1, b_mask_sts_1, b_ids_sts_2, b_mask_sts_2, b_labels_sts = (batch_sts['token_ids_1'], batch_sts['attention_mask_1'], 
                                                    batch_sts['token_ids_2'], batch_sts['attention_mask_2'], batch_sts['labels'])

            b_ids_sts_1 = b_ids_sts_1.to(device)
            b_ids_sts_2 = b_ids_sts_2.to(device)
            b_mask_sts_1 = b_mask_sts_1.to(device)
            b_mask_sts_2 = b_mask_sts_2.to(device)
            b_labels_sts = b_labels_sts.to(device)

            logits_sts = model.predict_similarity(b_ids_sts_1, b_mask_sts_1, b_ids_sts_2, b_mask_sts_2)
            loss_sts = F.mse_loss(logits_sts.view(-1,1), b_labels_sts.view(-1,1).float(), reduction='sum') / args.batch_size

            adv_loss_sts = adversarial_loss_sentence(model, b_ids_sts_1, b_mask_sts_1, b_ids_sts_2, b_mask_sts_2, b_labels_sts, which='STS')
            loss_sts += la_lambda * adv_loss_sts + l2_lambda * l2_reg

            optimizer.zero_grad()
            loss_sts.backward(retain_graph=True)
            optimizer.step()

            train_loss_sts += loss_sts.item()
            num_batches_sts += 1

        for batch_extra in tqdm(extra_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):    
            # Extra Similarity Task
            b_ids_extra_1, b_mask_extra_1, b_ids_extra_2, b_mask_extra_2, b_labels_extra = (batch_extra['token_ids_1'], batch_extra['attention_mask_1'], 
                                                    batch_extra['token_ids_2'], batch_extra['attention_mask_2'], batch_extra['labels'])

            b_ids_extra_1 = b_ids_extra_1.to(device)
            b_ids_extra_2 = b_ids_extra_2.to(device)
            b_mask_extra_1 = b_mask_extra_1.to(device)
            b_mask_extra_2 = b_mask_extra_2.to(device)
            b_labels_extra = b_labels_extra.to(device)

            print("IDS:", b_ids_extra_1)
            print("LABELS", b_labels_extra)

            logits_extra = model.predict_similarity(b_ids_extra_1, b_mask_extra_1, b_ids_extra_2, b_mask_extra_2)
            loss_extra = F.mse_loss(logits_extra.view(-1,1), b_labels_extra.view(-1,1).float(), reduction='sum') / args.batch_size

            adv_loss_extra = adversarial_loss_sentence(model, b_ids_extra_1, b_mask_extra_1, b_ids_extra_2, b_mask_extra_2, b_labels_extra, which='STS')
            loss_extra += la_lambda * adv_loss_extra + l2_lambda * l2_reg
            loss_extra *= 0.001

            optimizer.zero_grad()
            loss_extra.backward(retain_graph=True)
            optimizer.step()

            train_loss_extra += loss_extra.item()
            num_batches_extra += 1

        avg_train_loss_sst = train_loss_sst / num_batches_sst
        avg_train_loss_para = train_loss_para / num_batches_para
        avg_train_loss_sts = train_loss_sts / num_batches_sts
        avg_train_loss_extra = train_loss_extra / num_batches_extra

        print(avg_train_loss_sst, avg_train_loss_para, avg_train_loss_sts, avg_train_loss_extra)

        print(f"Training Set")
        paraphrase_accuracy, _, _, sentiment_accuracy, _, _, sts_corr, *_ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        print(f"Dev Set")
        dev_paraphrase_accuracy, _, _, dev_sentiment_accuracy, _, _, dev_sts_corr, *_ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        total_average = (dev_paraphrase_accuracy + dev_sentiment_accuracy + dev_sts_corr) / 3
        best_average = (best_sen_acc + best_para_acc + best_sts_corr) / 3

        print(f"Epoch {epoch}: sst train loss :: {avg_train_loss_sst :.3f}, sentiment acc :: {sentiment_accuracy :.3f}, dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
        print(f"Epoch {epoch}: para train loss :: {avg_train_loss_para :.3f}, para acc :: {paraphrase_accuracy :.3f}, dev para acc :: {dev_paraphrase_accuracy :.3f}")
        print(f"Epoch {epoch}: sts train loss :: {avg_train_loss_sts :.3f}, sts acc :: {sts_corr :.3f}, dev sts corr :: {dev_sts_corr :.3f}")

        # Early Stopping Implementation
        if (total_average > best_average):
            best_sen_acc = dev_sentiment_accuracy
            best_para_acc = dev_paraphrase_accuracy
            best_sts_corr = dev_sts_corr
        
            counter = 0
            save_model(model, optimizer, args, config, args.filepath)

        elif (counter >= 12):
            # End training to prevent overfitting
            break

        elif (counter >= 5):
            # Half Learning Rate
            lr = lr/2
            optimizer = AdamW(model.parameters(), lr=lr)
            counter += 1

        else:
            counter += 1
        
def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-best-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
