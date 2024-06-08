import torch
import random
import logging
import numpy as np
import torch.nn as nn
import networkx as nx
from tqdm import tqdm
import torch.optim as optim
import torchtext.vocab as vocab
import torch.nn.functional as F
from datasets import load_dataset
from itertools import combinations
from collections import defaultdict
from boltons.iterutils import pairwise, windowed
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

glove = vocab.GloVe()
dataset = load_dataset("conll2012_ontonotesv5", 'english_v4')

class CoNLLDatasetDocuments:
    def __init__(self, data):
        self.documents = []
        self.doc_ids = []
        self.max_len = 0
        self.pad_idx = len(glove.stoi)
        self.unk = torch.randn((1,300))
        for doc in data:
            sentences = doc['sentences']
            sentence_data = []
            for sentence in sentences:
                sentence_data.append({
                    'words': sentence['words'],
                    'speaker': sentence['speaker'],
                    'name_entities': sentence['named_entities'],
                    'coref_spans': sentence['coref_spans']
                })
                self.max_len = max(self.max_len, len(sentence_data[-1]['words']))
            self.documents.append(sentence_data)
            self.doc_ids.append(doc['document_id'])

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        return self.doc_ids[idx], self.documents[idx]

def flatten(alist):
    return [item for sublist in alist for item in sublist]

class Span:
    def __init__(self, start, end, id, speaker, genre):
        self.start = start
        self.end = end
        self.id = id
        self.speaker = speaker
        self.genre = genre

    def __len__(self):
        return self.end-self.start+1
    
class Document:
    def __init__(self, document_id, doc):
        self.doc = doc
        self.sentences = [sentence['words'] for sentence in doc]
        self.tokens = flatten(self.sentences)
        self.sentences_speakers = [sentence['speaker'] for sentence in doc]
        self.corefs = self.get_corefs()
        self.speakers = self.get_speakers()
        self.genre = document_id.split('/')[0]
        self.filename = document_id

    def __getitem__(self, idx):
        return (self.tokens[idx], self.corefs[idx], self.speakers[idx], self.genre)

    def __len__(self):
        return len(self.tokens)
    
    def get_corefs(self):
        # get gold corefs for the document with label, start and end index(inclusive)
        index = 0
        corefs = []
        for sentence in self.doc:
            for coref in sentence['coref_spans']:
                corefs.append({
                    'label': coref[0],
                    'start': coref[1] + index,
                    'end': coref[2] + index
                })
            index += len(sentence['words'])
        
        return corefs
    
    def get_speakers(self):
        speakers = []
        for sentence in self.doc:
            speakers.extend([sentence['speaker']]*len(sentence['words']))
        return speakers
    
    def compute_idx_spans(self, sentences, L=10):
        # Compute span indexes for all possible spans up to length L in each sentence 
        idx_spans = []
        shift = 0
        for sent in sentences:
            sent_spans = flatten([windowed(range(shift, len(sent)+shift), length)
                                for length in range(1, L)])
            idx_spans.extend(sent_spans)
            shift += len(sent)

        return idx_spans

    def spans(self):
        return [Span(start=i[0], end=i[-1], id=idx,
                    speaker=self.speaker(i), genre=self.genre)
                for idx, i in enumerate(self.compute_idx_spans(self.sentences))]

    def truncate(self, MAX=50):
        # Randomly truncate the document to up to MAX sentencess
        if len(self.sentences) > MAX:
            i = random.sample(range(MAX, len(self.sentences)), 1)[0]
            updated_doc = self.doc[i-MAX:i]
            return self.__class__(self.filename, updated_doc)
        return self

    def speaker(self, i):
        # Compute speaker of a span
        if self.speakers[i[0]] == self.speakers[i[-1]]:
            return self.speakers[i[0]]
        return None

class GloveModified:
    # GLove embeddings
    def __init__(self):
        self.glove = glove
        self.pad_t = 0
        self.unk_t = 1
        self.pad_emb = torch.zeros((1,300))
        self.unk_emb = torch.randn((1,300))
        self.vocab = ['<PAD>', '<UNK>'] + [v for v in self.glove.stoi]
        self.stoi = {s:i for i,s in enumerate(self.vocab)}
        self.itos = {i:s for s,i in self.stoi.items()}

    def __getitem__(self, idx):
        if idx == 0:
            return self.pad_emb
        elif idx == 1:
            return self.unk_emb
        else:
            return self.glove[self.itos[idx]]

    def __len__(self):
        return len(self.stoi)
    
class BertModified:
    # BERT pretrained embeddings
    def __init__(self):
        self.glove = glove
        self.bert_tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-cased')
        self.bert_model = BertModel.from_pretrained('google-bert/bert-base-cased')
        self.pad_t = 0
        self.unk_t = 1
        self.pad_emb = torch.zeros((1,300))
        self.unk_emb = torch.randn((1,300))
        self.vocab = ['<PAD>', '<UNK>'] + [v for v in self.glove.stoi]
        self.stoi = {s:i for i,s in enumerate(self.vocab)}
        self.itos = {i:s for s,i in self.stoi.items()}

    def __getitem__(self, idx):
        if idx == 0:
            return self.pad_emb
        elif idx == 1:
            return self.unk_emb
        else:
            sentences = self.itos[idx]
            inputs = self.bert_tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            bert_embeddings = outputs.last_hidden_state
            return bert_embeddings

    def __len__(self):
        return len(self.stoi)
        
    
class DocumentEncoder(nn.Module):
    # Document encoder for tokens
    def __init__(self, hidden_dim, n_layers=2):
        super().__init__()

        self.glove_m_ = GloveModified()

        vocab_size = len(self.glove_m_)
        embedding_dim = 300 

        self.glove_m = nn.Embedding(vocab_size, embedding_dim)

        embedding_weights = torch.zeros(vocab_size, embedding_dim)
        for idx, word in self.glove_m_.itos.items():
            embedding_weights[idx] = self.glove_m_[idx]

        self.glove_m.weight.data.copy_(embedding_weights)
        self.glove_m.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True)

        self.emb_dropout = nn.Dropout(0.50, inplace=True)
        self.lstm_dropout = nn.Dropout(0.20, inplace=True)

    def forward(self, doc):
        # Convert document words to ids, embed them, pass through LSTM

        embeds = [self.embed(s) for s in doc.sentences]

        sizes = [t.shape[0] for t in embeds]
        pad_embeds = pad_sequence(embeds, batch_first=True)
        packed = pack_padded_sequence(pad_embeds, sizes, batch_first=True, enforce_sorted=False)

        packed_data, packed_batch_sizes = packed.data, packed.batch_sizes
        packed_data = self.emb_dropout(packed_data)
        packed = torch.nn.utils.rnn.PackedSequence(packed_data, packed_batch_sizes)

        output, _ = self.lstm(packed)

        output_data = self.lstm_dropout(output.data)
        output = torch.nn.utils.rnn.PackedSequence(output_data, output.batch_sizes)

        unpacked, sizes = pad_packed_sequence(output, batch_first=True)

        outputs_unpadded = [unpacked[idx][:val] for idx, val in enumerate(sizes)]

        return torch.cat(outputs_unpadded, dim=0), torch.cat(embeds, dim=0)
    def embed(self, sent):
        # Embed a sentence using GLoVE/BERT embeddings
        glove_embeds = self.glove_m(torch.tensor([self.glove_m_.stoi[token] if token in self.glove_m_.stoi else self.glove_m_.stoi['<UNK>'] for token in sent]))

        return glove_embeds


class mlp(nn.Module):
    # Scoring mlp, architecture from the paper
    def __init__(self, embeds_dim, hidden_dim=150):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(embeds_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):

        return self.score(x)

class phi_distance(nn.Module):
    # Learned, continuous representations for: span widths, distance between spans
    def __init__(self, distance_dim=20):
        super().__init__()
        self.bins = [1,2,3,4,5,8,16,32,64]
        self.dim = distance_dim
        self.embeds = nn.Embedding(len(self.bins)+1, distance_dim)

    def forward(self, lengths):
        bin_tensor = self.length_to_bin(lengths)
        return self.embeds(bin_tensor)

    def length_to_bin(self, lengths):
        # Find which bin a number falls into
        return torch.tensor([sum([True for i in self.bins if num >= i]) for num in lengths], requires_grad=False)

class MentionScore(nn.Module):
    # Mention scoring module
    def __init__(self, gi_dim, lstm_out_dim, distance_dim, wctf = None):
        super().__init__()

        self.attention = mlp(lstm_out_dim)
        self.distance_embedding = phi_distance(distance_dim)
        self.mention_score = mlp(gi_dim)
        if wctf is not None:
            self.ctf = True
            self.wctf = wctf
        else:
            self.ctf = False

    def prune(self, spans, T, LAMBDA=0.40):
        # Prune mention scores to the top lambda percent.Returns list of tuple(scores, indices, g_i)

        # Only take top λT spans, where T = len(doc)
        STOP = int(LAMBDA * T)

        # Sort by mention score, remove overlapping spans, prune to top λT spans
        sorted_spans = sorted(spans, key=lambda s: s.si, reverse=True)
        nonoverlapping = self.remove_overlapping(sorted_spans)
        pruned_spans = nonoverlapping[:STOP]

        # Resort by start, end indexes
        spans = sorted(pruned_spans, key=lambda s: (s.start, s.end))

        return spans

    def remove_overlapping(self, sorted_spans):
        # Remove spans that are overlapping by order of decreasing mention score
        nonoverlapping, seen = [], set()
        for s in sorted_spans:
            indexes = range(s.start, s.end+1)
            taken = [i in seen for i in indexes]
            if len(set(taken)) == 1 or (taken[0] == taken[-1] == False):
                nonoverlapping.append(s)
                seen.update(indexes)

        return nonoverlapping

    def forward(self, states, embeds, doc, k = 250):

        spans = doc.spans()

        # Compute first part of attention over span states (alpha_t)
        attns = self.attention(states)

        span_attns = [attns[s.start:s.end+1] for s in spans]
        span_embeds = [embeds[s.start:s.end+1] for s in spans]

        padded_attns = pad_sequence(span_attns, batch_first=True, padding_value=-1e10)
        padded_embeds = pad_sequence(span_embeds, batch_first=True, padding_value=0)

        attn_weights = F.softmax(padded_attns, dim=1)

        attn_embeds = torch.sum(torch.mul(padded_embeds, attn_weights), dim=1)

        widths = self.distance_embedding([len(s) for s in spans])

        start_end = torch.stack([torch.cat((states[s.start], states[s.end])) for s in spans])

        g_i = torch.cat((start_end, attn_embeds, widths), dim=1)

        mention_scores = self.mention_score(g_i)

        spans_score_updated = []
        for span, si in zip(spans, mention_scores.detach()):
            span.si = si
            spans_score_updated.append(span)
        
        spans = spans_score_updated

        spans = self.prune(spans, len(doc))

        spans_antecedents_updated = []
        for idx, span in enumerate(spans):
            if self.ctf:
                scores = torch.tensor([mention_scores[span.id] + mention_scores[s.id] + g_i[span.id]@self.wctf@g_i[s.id] for s in spans[:idx]])
                sorted_span_index = torch.argsort(scores, descending=True)
                span.yi = [spans[i] for i in sorted_span_index]

            else:
                span.yi = spans[max(0, idx-k):idx]
            spans_antecedents_updated.append(span)

        spans = spans_antecedents_updated
        

        return spans, g_i, mention_scores

class Genre(nn.Module):
    # Representations for genre. Zeros if genre unknown
    def __init__(self, genre_dim=20):
        super().__init__()
        self.genres = ['bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb']
        self.g_to_i = {genre: idx+1 for idx, genre in enumerate(self.genres)}

        self.embeds = nn.Embedding(len(self.genres)+1, genre_dim, padding_idx=0)

    def forward(self, labels):
        bin_tensor = self.g_to_i_lookup(labels)
        return self.embeds(bin_tensor)

    def g_to_i_lookup(self, labels):
        indexes = [self.g_to_i[gen] if gen in self.g_to_i else None for gen in labels]
        return torch.tensor([i if i is not None else 0 for i in indexes], requires_grad=False)


class Speaker(nn.Module):
    # Learned continuous representations for binary speaker. Zeros if speaker unknown

    def __init__(self, speaker_dim=20):
        super().__init__()

        self.embeds = nn.Embedding(3, speaker_dim, padding_idx=0)

    def forward(self, speaker_labels):
        return self.embeds(torch.tensor(speaker_labels))

class PairwiseScore(nn.Module):
    # Coreference pair scoring module
    def __init__(self, gi_dim, gij_dim, distance_dim, genre_dim, speaker_dim, N = 1, wctf = None):
        super().__init__()

        self.distance_embedding = phi_distance(distance_dim)
        self.genre_embedding = Genre(genre_dim)
        self.speaker_embedding = Speaker(speaker_dim)

        self.pairwise_score = mlp(gij_dim)

        self.order = N

        self.fmatrix = nn.Linear(2*gi_dim, 1)

        if wctf is not None:
            self.ctf = True
            self.wctf = wctf
        else:
            self.ctf = False

    def compare_span_speaker(self, s1, s2):

        if s1.speaker == s2.speaker:
            idx = torch.tensor(1)

        elif s1.speaker != s2.speaker:
            idx = torch.tensor(2)

        else:
            idx = torch.tensor(0)

        return idx
    
    def pairwise_indexes(self, spans):
        # Get indices for indexing into pairwise_scores
        indexes = [0] + [len(s.yi) for s in spans]
        indexes = [sum(indexes[:idx+1]) for idx, _ in enumerate(indexes)]
        return pairwise(indexes)
    
    def higher_order_refinement(self, g_i, mention_ids, antecedent_ids, pair_scores, span_antecedents_len):
        # Refine span representations for Higher order resolution
        j_g = torch.index_select(g_i, 0, antecedent_ids)
        pair_scores = pair_scores.flatten()
        g_i_new = g_i.clone()

        index = 0

        for mention_len in span_antecedents_len:
            if mention_len == 0:
                continue
            scores = pair_scores[index:index+mention_len]
            attn_weights = F.softmax(scores, dim=0).unsqueeze(1)
            weighted_g = torch.sum(attn_weights*j_g[index:index+mention_len], dim = 0)
            mention_id = mention_ids[index:index+mention_len][0]
            mention_g = g_i[mention_id]
            f = F.sigmoid(self.fmatrix(torch.cat((mention_g, weighted_g), dim = 0)))

            g_i_new[mention_id] = f*mention_id + (1-f)*weighted_g

            index += mention_len

        return g_i_new     

    def forward(self, spans, g_i, mention_scores):
        # Compute pairwise score for spans and their up to K antecedents

        mention_ids, antecedent_ids , distances, genres , speakers = [], [], [], [], []

        span_antecedents_len = []

        for i in spans:
            span_antecedents_len.append(len(i.yi))
            for j in i.yi:
                mention_ids.append(i.id)
                antecedent_ids.append(j.id)
                distances.append(i.end - j.start)
                genres.append(i.genre)
                speakers.append(self.compare_span_speaker(i, j))
        
        mention_ids = torch.tensor(mention_ids)
        antecedent_ids = torch.tensor(antecedent_ids)


        phi = torch.cat((self.distance_embedding(distances), self.genre_embedding(genres), self.speaker_embedding(speakers)), dim=1)

        for n in range(self.order):
            i_g = torch.index_select(g_i, 0, mention_ids)
            j_g = torch.index_select(g_i, 0, antecedent_ids)

            # if course to fine, 
            if self.ctf:
                s_c = torch.sum((i_g@self.wctf)*i_g, dim = 1, keepdim = True)

            pairs = torch.cat((i_g, j_g, i_g*j_g, phi), dim=1)

            s_i = torch.index_select(mention_scores, 0, mention_ids)
            s_j = torch.index_select(mention_scores, 0, antecedent_ids)

            pairwise_scores = self.pairwise_score(pairs)
            
            if n < self.order-1:
                g_i = self.higher_order_refinement(g_i, mention_ids, antecedent_ids, pairwise_scores, span_antecedents_len)

        if self.ctf:
            coref_scores = torch.sum(torch.cat((s_i, s_j, s_c, pairwise_scores), dim=1), dim=1, keepdim=True)
        else:
            coref_scores = torch.sum(torch.cat((s_i, s_j, pairwise_scores), dim=1), dim=1, keepdim=True)

        spans_antecedents_idx_updated = []
        for span in spans:
            span.yi_idx = [((y.start, y.end), (span.start, span.end)) for y in span.yi]
            spans_antecedents_idx_updated.append(span)

        spans = spans_antecedents_idx_updated
        antecedent_idx = [len(s.yi) for s in spans if len(s.yi)]

        split_scores = [torch.tensor([])] + list(torch.split(coref_scores, antecedent_idx, dim=0))

        epsilon = torch.tensor([[0.]])
        with_epsilon = [torch.cat((score, epsilon), dim=0) for score in split_scores]

        probs = [F.softmax(tens, dim = 0) for tens in with_epsilon]
        
        probs = pad_sequence(probs, batch_first= True, padding_value=69)
        probs = probs.squeeze()
       
        return spans, probs


class CorefScore(nn.Module):
    # Super class to compute coreference links between spans
    def __init__(self, embeds_dim, hidden_dim, distance_dim=20, genre_dim=20, speaker_dim=20, ctf = False):

        super().__init__()

        lstm_out_dim = hidden_dim*2
        distance_dim = distance_dim
        genre_dim = genre_dim
        speaker_dim = speaker_dim
        token_emb_dim = embeds_dim
        gi_dim = 2*lstm_out_dim + token_emb_dim + distance_dim
        gij_dim = gi_dim*3 + distance_dim + genre_dim + speaker_dim
        if ctf:
            wctf = nn.Parameter(torch.randn((gi_dim, gi_dim)))
        else:
            wctf = None

        self.document_encoder = DocumentEncoder(hidden_dim)
        self.spans_mention_score = MentionScore(gi_dim, lstm_out_dim, distance_dim, wctf)
        self.spans_pairwise_score = PairwiseScore(gi_dim, gij_dim, distance_dim, genre_dim, speaker_dim, N = 1, wctf = wctf)

    def forward(self, doc):
        # Encode the document, keep the LSTM hidden states and embedded tokens
        states, embeds = self.document_encoder(doc)

        # Get mention scores for each span, prune
        spans, g_i, mention_scores = self.spans_mention_score(states, embeds, doc)

        # Get pairwise scores for each span combo
        spans, coref_scores = self.spans_pairwise_score(spans, g_i, mention_scores)

        return spans, coref_scores

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)

class Trainer:
    """ Class dedicated to training and evaluating the model"""
    def __init__(self, model, train_corpus, val_corpus, lr=1e-3, batch_size = 100):

        self.train_corpus = train_corpus
        self.val_corpus = val_corpus
        self.batch_size = batch_size
        
        self.model = model

        self.optimizer = optim.Adam(params=[p for p in self.model.parameters() if p.requires_grad],lr=lr)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,step_size=100,gamma=0.999)
    
    def modified_divide(self, x, y):
        if y != 0:
            return x / y
        return 1

    def train(self, num_epochs, eval_interval=10):
        """ Train a model """
        for epoch in range(1, num_epochs+1):
            """ Run a training epoch over 'steps' documents """

            self.model.train()

            batch = random.sample(self.train_corpus, self.batch_size)

            epoch_loss, epoch_mentions, epoch_corefs, epoch_identified = [], [], [], []

            for document in tqdm(batch, desc=f'Training epoch {epoch}'):

                doc = document.truncate()

                loss, mentions_found, total_mentions, corefs_found, total_corefs, corefs_chosen = self.train_doc(doc)

                logger.info(f'{document.filename} | Loss: {loss} | Mentions: {mentions_found}/{total_mentions} | Coref recall: {corefs_found}/{total_corefs} | Corefs precision: {corefs_chosen}/{total_corefs}')

                epoch_loss.append(loss)
                epoch_mentions.append(self.modified_divide(mentions_found, total_mentions))
                epoch_corefs.append(self.modified_divide(corefs_found, total_corefs))
                epoch_identified.append(self.modified_divide(corefs_chosen, total_corefs))

            self.scheduler.step()

            logger.info(f'Epoch: {epoch} | Loss: {np.mean(epoch_loss)} | Mention recall: {np.mean(epoch_mentions)} | Coref recall: {np.mean(epoch_corefs)} | Coref precision: {np.mean(epoch_identified)}')

            if epoch % eval_interval == 0:
                torch.save(self.model.state_dict(), f'model_epoch_{epoch}.pt')

    def extract_gold_corefs(self, document):
        #  Parse coreference dictionary of a document to get coref links

        gold_links = defaultdict(list)

        gold_mentions = set([(coref['start'],coref['end']) for coref in document.corefs])
        total_mentions = len(gold_mentions)

        for coref_entry in document.corefs:

            label =  coref_entry['label']
            span_idx  = (coref_entry['start'],coref_entry['end'])

            gold_links[label].append(span_idx) 

        gold_corefs = flatten([[coref for coref in combinations(gold, 2)] for gold in gold_links.values()])
        gold_corefs = sorted(gold_corefs)
        total_corefs = len(gold_corefs)

        return gold_corefs, total_corefs, gold_mentions, total_mentions

    def train_doc(self, document):
        #  Compute loss for a forward pass over a document

        gold_corefs, total_corefs, gold_mentions, total_mentions = self.extract_gold_corefs(document)

        self.optimizer.zero_grad()

        mentions_found, corefs_found, corefs_chosen = 0, 0, 0

        spans, probs = self.model(document)

        gold_indexes = torch.zeros_like(probs)
        for idx, span in enumerate(spans):

            if (span.start, span.end) in gold_mentions:
                mentions_found += 1
                golds = [i for i, link in enumerate(span.yi_idx) if link in gold_corefs]

                if golds:
                    gold_indexes[idx, golds] = 1

                    corefs_found += len(golds)
                    found_corefs = sum((probs[idx, golds] > probs[idx, len(span.yi_idx)])).detach()
                    corefs_chosen += found_corefs.item()
                else:
                    gold_indexes[idx, len(span.yi_idx)] = 1

        # Negative marginal log-likelihood
        eps = 1e-8
        loss = torch.sum(torch.log(torch.sum(torch.mul(probs, gold_indexes), dim=1).clamp_(eps, 1-eps)) * -1) / len(probs)

        loss.backward()

        self.optimizer.step()

        return (loss.item(), mentions_found, total_mentions, corefs_found, total_corefs, corefs_chosen)
    
    def predict(self, document):
        # Predict coreference clusters in a document
        self.model.eval()
        graph = nx.Graph()
        spans, probs = self.model(document)

        for i, span in enumerate(spans):

            found_corefs = [idx for idx, _ in enumerate(span.yi_idx) if probs[i, idx] > probs[i, len(span.yi_idx)]]

            if any(found_corefs):
                for coref_idx in found_corefs:
                    link = spans[coref_idx]
                    graph.add_edge((span.start, span.end), (link.start, link.end))

        clusters = list(nx.connected_components(graph))

        for idx, cluster in enumerate(clusters):
            print(f"Cluster - {idx+1}:")
            for start, end in cluster:
                print(" ".join([document.tokens[i] for i in range(start, end+1)]))
            print("********************************************************************")

        return document

logger.info("Loadind datasets")
train_documents = CoNLLDatasetDocuments(dataset['train'])
val_documents = CoNLLDatasetDocuments(dataset['validation'])
train_documents = [Document(id, doc) for id, doc in train_documents]
val_documents = [Document(id, doc) for id, doc in val_documents]

logger.info("Initialising Model")
model = CorefScore(embeds_dim=300, hidden_dim=200, ctf=True)

logger.info("Training Started")
trainer = Trainer(model, train_documents, val_documents, batch_size = 100)
trainer.train(150)




