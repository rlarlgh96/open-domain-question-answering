import json
import os
import pickle
import time
import random
from contextlib import contextmanager
from typing import Callable, List, NoReturn, Optional, Tuple, Union

from arguments import DataTrainingArguments
import faiss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    concatenate_datasets, 
    load_from_disk
)
from transformers import (
    AutoTokenizer,
    AutoModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
)
from transformers import TrainingArguments
from encoder import BertEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi, BM25L, BM25Plus

seed = 2024
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn, ngram_range=(1, 2), max_features=50000,
        )

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

    def get_sparse_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"sparse_embedding.bin"
        tfidfv_name = f"tfidv.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")

    # def build_faiss(self, num_clusters=64) -> NoReturn:

    #     """
    #     Summary:
    #         속성으로 저장되어 있는 Passage Embedding을
    #         Faiss indexer에 fitting 시켜놓습니다.
    #         이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

    #     Note:
    #         Faiss는 Build하는데 시간이 오래 걸리기 때문에,
    #         매번 새롭게 build하는 것은 비효율적입니다.
    #         그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
    #         다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
    #         제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
    #     """

    #     indexer_name = f"faiss_clusters{num_clusters}.index"
    #     indexer_path = os.path.join(self.data_path, indexer_name)
    #     if os.path.isfile(indexer_path):
    #         print("Load Saved Faiss Indexer.")
    #         self.indexer = faiss.read_index(indexer_path)

    #     else:
    #         p_emb = self.p_embedding.astype(np.float32).toarray()
    #         emb_dim = p_emb.shape[-1]

    #         num_clusters = num_clusters
    #         quantizer = faiss.IndexFlatL2(emb_dim)

    #         self.indexer = faiss.IndexIVFScalarQuantizer(
    #             quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
    #         )
    #         self.indexer.train(p_emb)
    #         self.indexer.add(p_emb)
    #         faiss.write_index(self.indexer, indexer_path)
    #         print("Faiss Indexer Saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        with timer("transform"):
            query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                여러 개의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    # def retrieve_faiss(
    #     self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    # ) -> Union[Tuple[List, List], pd.DataFrame]:

    #     """
    #     Arguments:
    #         query_or_dataset (Union[str, Dataset]):
    #             str이나 Dataset으로 이루어진 Query를 받습니다.
    #             str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
    #             Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
    #             이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
    #         topk (Optional[int], optional): Defaults to 1.
    #             상위 몇 개의 passage를 사용할 것인지 지정합니다.

    #     Returns:
    #         1개의 Query를 받는 경우  -> Tuple(List, List)
    #         다수의 Query를 받는 경우 -> pd.DataFrame: [description]

    #     Note:
    #         다수의 Query를 받는 경우,
    #             Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
    #             Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
    #         retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
    #     """

    #     assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

    #     if isinstance(query_or_dataset, str):
    #         doc_scores, doc_indices = self.get_relevant_doc_faiss(
    #             query_or_dataset, k=topk
    #         )
    #         print("[Search query]\n", query_or_dataset, "\n")

    #         for i in range(topk):
    #             print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
    #             print(self.contexts[doc_indices[i]])

    #         return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

    #     elif isinstance(query_or_dataset, Dataset):

    #         # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
    #         queries = query_or_dataset["question"]
    #         total = []

    #         with timer("query faiss search"):
    #             doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
    #                 queries, k=topk
    #             )
    #         for idx, example in enumerate(
    #             tqdm(query_or_dataset, desc="Sparse retrieval: ")
    #         ):
    #             tmp = {
    #                 # Query와 해당 id를 반환합니다.
    #                 "question": example["question"],
    #                 "id": example["id"],
    #                 # Retrieve한 Passage의 id, context를 반환합니다.
    #                 "context": " ".join(
    #                     [self.contexts[pid] for pid in doc_indices[idx]]
    #                 ),
    #             }
    #             if "context" in example.keys() and "answers" in example.keys():
    #                 # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
    #                 tmp["original_context"] = example["context"]
    #                 tmp["answers"] = example["answers"]
    #             total.append(tmp)

    #         return pd.DataFrame(total)

    # def get_relevant_doc_faiss(
    #     self, query: str, k: Optional[int] = 1
    # ) -> Tuple[List, List]:

    #     """
    #     Arguments:
    #         query (str):
    #             하나의 Query를 받습니다.
    #         k (Optional[int]): 1
    #             상위 몇 개의 Passage를 반환할지 정합니다.
    #     Note:
    #         vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
    #     """

    #     query_vec = self.tfidfv.transform([query])
    #     assert (
    #         np.sum(query_vec) != 0
    #     ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

    #     q_emb = query_vec.toarray().astype(np.float32)
    #     with timer("query faiss search"):
    #         D, I = self.indexer.search(q_emb, k)

    #     return D.tolist()[0], I.tolist()[0]

    # def get_relevant_doc_bulk_faiss(
    #     self, queries: List, k: Optional[int] = 1
    # ) -> Tuple[List, List]:

    #     """
    #     Arguments:
    #         queries (List):
    #             하나의 Query를 받습니다.
    #         k (Optional[int]): 1
    #             상위 몇 개의 Passage를 반환할지 정합니다.
    #     Note:
    #         vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
    #     """

    #     query_vecs = self.tfidfv.transform(queries)
    #     assert (
    #         np.sum(query_vecs) != 0
    #     ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

    #     q_embs = query_vecs.toarray().astype(np.float32)
    #     D, I = self.indexer.search(q_embs, k)

    #     return D.tolist(), I.tolist()

class DenseRetrieval:

    def __init__(
        self, 
        args,
        model_name,
        train_dataset,
        eval_dataset,
        num_neg,
        model_path="./models/encoder",
        data_path="../data",
        context_path="wikipedia_documents.json",
    ):

        '''
        학습과 추론에 사용될 여러 셋업을 마쳐봅시다.
        '''

        self.args = args
        self.model_name = model_name
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.num_neg = num_neg
        self.model_path = model_path
        self.data_path = data_path

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)
        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.p_encoder = BertEncoder.from_pretrained(model_name)
        self.q_encoder = BertEncoder.from_pretrained(model_name)

        with open(os.path.join(self.data_path, "sparse_embedding.bin"), "rb") as file:
            self.p_embedding = pickle.load(file)
        with open(os.path.join(self.data_path, "tfidv.bin"), "rb") as file:
            self.tfidfv = pickle.load(file)

    def get_dense_embedding(self) -> NoReturn:
        passage_seqs = self.tokenizer(self.contexts, padding="max_length", truncation=True, return_tensors='pt')
        passage_dataset = TensorDataset(
            passage_seqs['input_ids'], passage_seqs['attention_mask'], passage_seqs['token_type_ids']
        )
        self.passage_dataloader = DataLoader(passage_dataset, batch_size=self.args.per_device_train_batch_size)

        p_encoder_path = os.path.join(self.model_path, 'p_encoder.pth')
        q_encoder_path = os.path.join(self.model_path, 'q_encoder.pth')

        if os.path.isfile(p_encoder_path) and os.path.isfile(q_encoder_path):
            self.p_encoder.load_state_dict(torch.load(p_encoder_path))
            self.q_encoder.load_state_dict(torch.load(q_encoder_path))
            print("Encoder loaded.")
        else:
            print("Train encoder.")
            self.prepare_in_batch_negative(num_neg=self.num_neg)
            self.p_encoder, self.q_encoder = self.train()
            torch.save(self.p_encoder.state_dict(), p_encoder_path)
            torch.save(self.q_encoder.state_dict(), q_encoder_path)
            print("Encoder saved.")

    def prepare_in_batch_negative(self, train_dataset=None, contexts=None, num_neg=2, tokenizer=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        if tokenizer is None:
            tokenizer = self.tokenizer

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.
        # 1) tfidfv score
        corpus = self.contexts
        p_with_neg = []

        with open(os.path.join(self.data_path, "sparse_embedding.bin"), "rb") as file:
            p_embedding = pickle.load(file)
        with open(os.path.join(self.data_path, "tfidv.bin"), "rb") as file:
            tfidfv = pickle.load(file)

        for p, q, a in zip(train_dataset['context'], train_dataset['question'], train_dataset['answers']):
            p_with_neg.append(p)

            q_vec = tfidfv.transform([q])
            result = q_vec * p_embedding.T
            result = result.toarray()
            sorted_result = np.argsort(result.squeeze())[::-1]
            doc_indices = sorted_result.tolist()

            cnt = 0
            for idx in doc_indices:
                if not a['text'][0] in corpus[idx]:
                    p_with_neg.append(corpus[idx])
                    cnt += 1
                else:
                    continue

                if cnt == num_neg:
                    break

        # 2) random
        # corpus = np.array(self.contexts)
        # p_with_neg = []

        # for c in train_dataset['context']:

        #     while True:
        #         neg_idxs = np.random.randint(len(corpus), size=num_neg)

        #         if not c in corpus[neg_idxs]:
        #             p_neg = corpus[neg_idxs]

        #             p_with_neg.append(c)
        #             p_with_neg.extend(p_neg)
        #             # print(p_with_neg)
        #             break

        # 3) BM25
        # corpus = self.contexts
        # p_with_neg = []
        # tokenized_corpus = [self.tokenizer(context) for context in self.contexts]
        # bm25 = BM25Plus(tokenized_corpus)          

        # for p, q, a in zip(train_dataset['context'], train_dataset['question'], train_dataset['answers']):
        #     p_with_neg.append(p)

        #     tokenized_query = self.tokenizer(q)
        #     result = bm25.get_scores(tokenized_query)
        #     sorted_result = np.argsort(result.squeeze())[::-1]
        #     doc_indices = sorted_result.tolist()

        #     cnt = 0
        #     for idx in doc_indices:
        #         if not a['text'][0] in corpus[idx]:
        #             p_with_neg.append(corpus[idx])
        #             cnt += 1
        #         else:
        #             continue

        #         if cnt == num_neg:
        #             break


        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(train_dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')

        max_len = p_seqs['input_ids'].size(-1)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

        dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        self.train_dataloader = DataLoader(dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size)

    def train(self, args=None):
        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # Start training!
        self.p_encoder = self.p_encoder.to(args.device)
        self.q_encoder = self.p_encoder.to(args.device)

        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        # train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        for _ in range(int(args.num_train_epochs)):
        # for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:
                    self.p_encoder.train()
                    self.q_encoder.train()
                    
                    batch = tuple(b.to(args.device) for b in batch)
                    targets = torch.zeros(batch_size).long().to(args.device) # positive example은 전부 첫 번째에 위치하므로

                    p_inputs = {
                        'input_ids': batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'attention_mask': batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'token_type_ids': batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                    }

                    q_inputs = {
                        'input_ids': batch[3].to(args.device),
                        'attention_mask': batch[4].to(args.device),
                        'token_type_ids': batch[5].to(args.device)
                    }

                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f'{str(loss.item())}')

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs
    
        return self.p_encoder, self.q_encoder

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query, k=1, args=None, p_encoder=None, q_encoder=None):
        if args is None:
            args = self.args
        if p_encoder is None:
            p_encoder = self.p_encoder
        if q_encoder is None:
            q_encoder = self.q_encoder

        p_encoder = p_encoder.to(args.device)
        q_encoder = q_encoder.to(args.device)

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to(args.device)
            q_emb = q_encoder(**q_seqs_val).to('cpu') # (num_query=1, emb_dim)

            p_embs = []
            for batch in self.passage_dataloader:
                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                p_emb = p_encoder(**p_inputs).to('cpu')
                p_embs.append(p_emb)

        p_embs = torch.cat(p_embs, dim=0)  # (num_passage, emb_dim)
        result = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        result = result.numpy()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices
    
    def get_relevant_doc_bulk(self, queries, k=1, args=None, p_encoder=None, q_encoder=None):
        if args is None:
            args = self.args
        if p_encoder is None:
            p_encoder = self.p_encoder
        if q_encoder is None:
            q_encoder = self.q_encoder

        p_encoder = p_encoder.to(args.device)
        q_encoder = q_encoder.to(args.device)

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors='pt').to(args.device)
            q_emb = q_encoder(**q_seqs_val).to('cpu') # (num_query=1, emb_dim)

            p_embs = []
            for batch in self.passage_dataloader:
                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                p_emb = p_encoder(**p_inputs).to('cpu')
                p_embs.append(p_emb)

        p_embs = torch.cat(p_embs, dim=0)  # (num_passage, emb_dim)
        result = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        result = result.numpy()

        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(
        tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
    )
    retriever.get_sparse_embedding()

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets

def run_dense_retrieval(
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "../data",
    train_path: str = "../data/train_dataset",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    args = TrainingArguments(
    output_dir="./models/encoder",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    )

    train_dataset = load_from_disk(train_path)["train"]
    eval_dataset = load_from_disk(train_path)["validation"]
    model_name = "klue/bert-base"

    retriever = DenseRetrieval(
        args=args, model_name=model_name, train_dataset=train_dataset, eval_dataset=eval_dataset, num_neg=7,
    )
    retriever.get_dense_embedding()

    if data_args.use_faiss:
        pass
        # retriever.build_faiss(num_clusters=data_args.num_clusters)
        # df = retriever.retrieve_faiss(
        #     datasets["validation"], topk=data_args.top_k_retrieval
        # )
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets