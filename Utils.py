import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from math import ceil
from typing import Callable
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
import openai
import gensim.downloader as api
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from time import sleep

import os
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM, AutoConfig
import shutil

    
def load_model(path,dtype):
    if path[:7] =="llama-2":
        model_save = "/tmp/"+path+"/snapshots/"
        print(model_save)
        model_save = model_save + os.listdir(model_save)[0]
        print(model_save)
    else:
        model_save = "/tmp/"+path
    if dtype == "float16":
        model = AutoModelForCausalLM.from_pretrained(model_save,torch_dtype=torch.float16,low_cpu_mem_usage=True,device_map="auto")
    elif dtype == "8bit":
        model= AutoModelForCausalLM.from_pretrained(model_save,load_in_8bit=True,low_cpu_mem_usage=True,device_map="auto")
    elif dtype == "4bit":
        model = AutoModelForCausalLM.from_pretrained(model_save,load_in_4bit=True,low_cpu_mem_usage=True,device_map="auto")
    elif dtype == "float32":
        model = AutoModelForCausalLM.from_pretrained(model_save,low_cpu_mem_usage=True,device_map="auto")
    else:
        assert False, "Dtype " + dtype + " not supported"
    return model

def initialize_models(paths,dtypes):
    models = {}
    for path,dtype in zip(paths,dtypes):
        print("Loading model: " + path)
        try:
            models[path+"_"+dtype] = load_model(path,dtype)   
        except:
            shutil.copytree("/fast/tsuehr/"+path,"/tmp/"+path)
            models[path+"_"+dtype] = load_model(path,dtype)
    return models
    

def get_tests(test):
    if test == "test_small":
        tests = {"test_small": pd.read_csv("psytests/test_small.csv", sep=';')}
    if test == "bfi":
        tests = {"bfi": pd.read_csv("psytests/BFI2.csv", sep=';')}
    elif test == "ipip":
        tests = {"neo120": pd.read_csv("psytests/NEO120.csv", sep=',')}
    elif test == "bfi_reverse":
        tests = {"bfi_reverse": pd.read_csv("psytests/NEO120_reverse.csv", sep=';')}
    elif test == "ipip_reverse":
        tests = {"neo120_reverse": pd.read_csv("psytests/BFI2_reverse.csv", sep=';')}
    elif test == "openpsych":
        tests = {"openpsych": pd.read_csv("psytests/Openpsych.csv", sep=';')}
    elif test == "original":
        tests = {"bfi": pd.read_csv("psytests/BFI2.csv", sep=';'),
        "neo120": pd.read_csv("psytests/NEO120.csv", sep=',')}
    elif test == "reverse":
        tests = {"bfi_reverse": pd.read_csv("psytests/BFI2_reverse.csv", sep=';'),
        "neo120_reverse": pd.read_csv("psytests/NEO120_reverse.csv", sep=';')}
    elif test == "all_old":
        tests = {"bfi": pd.read_csv("psytests/BFI2.csv", sep=';'),
        "neo120": pd.read_csv("psytests/NEO120.csv", sep=','),
        "bfi_reverse": pd.read_csv("psytests/BFI2_reverse.csv", sep=';'),
        "neo120_reverse": pd.read_csv("psytests/NEO120_reverse.csv", sep=';')}
    elif test == "all":
        tests = {"bfi": pd.read_csv("psytests/BFI2.csv", sep=';'),
        "neo120": pd.read_csv("psytests/NEO120.csv", sep=','),
        "bfi_reverse": pd.read_csv("psytests/BFI2_reverse.csv", sep=';'),
        "neo120_reverse": pd.read_csv("psytests/NEO120_reverse.csv", sep=';'),
        "openpsych": pd.read_csv("psytests/Openpsych.csv", sep=';')}
    return tests


def coagulator_default(meta,main_text,model):
    return  meta + "\n"  + main_text 

def coagulator_chat(meta,main_text,model):
    if type(model) == str and model in ["gpt-4-0613","gpt-4-0314","gpt-3.5-turbo-0613","gpt-3.5-turbo-0301"]:
        return [{"role": "system", "content":meta},{"role":"user","content":main_text}]
    elif type(model) == str and model in ["text-davinci-003","text-davinci-002","text-davinci-001","davinci"]:
        return meta + "\n" + main_text
    else:
        return "[INST] <<SYS>> \n " + meta + "\n <</SYS>> \n\n"  + main_text + "[/INST] "

def get_conditions(conditions):
    if conditions == "test_bfi":
        cond_options = list(zip([["1","2","3","4","5"]],
        ["ABC"]))

        cond_answers = list(zip([["Disagree strongly", "Disagree a little", "Neutral; no opinion", "Agree a little", "Agree strongly"],],
        ["Disagree Strongly to Agree Strongly"]))

        cond_p_answers = list(zip([[0,1,2,3,4],[4,3,2,1,0]],
        ["Default","Reverse"]))

        cond_p_options = list(zip([[0,1,2,3,4],[4,3,2,1,0]],
        ["Default","Reverse"]))

        cond_instructions = list(zip([lambda x: """Please indicate the extent to which you agree or disagree with the following statement: "I am someone who """ + x + """ "."""[1]],
        ["BFI base"]))

        cond_meta_instructions = list(zip([lambda x: "For the following task, respond in a way that matches this description: " + x + "\n Please respond with the single letter or number that represents your answer.",],
        ["Letter only"]))
    
        cond_coagulator = list(zip([coagulator_chat],
        ["Chat"]))

        cond_persona = ["i love to hang out with my friends. i love playing sports and being active. i am a 22 year old girl. i am in college studying education. i love rap music."]


    elif conditions == "personas_minimal":

        cond_options = list(zip([["1","2","3","4","5"]],
        ["123"]))

        cond_answers = list(zip([["Disagree", "Slightly disagree", "Neutral", "Slightly agree", "Agree"],],
        ["Disagree to Agree"]))

        cond_p_answers = list(zip([[0,1,2,3,4]],
        ["Default"]))

        cond_p_options = list(zip([[0,1,2,3,4]],
        ["Default"]))

        cond_instructions = list(zip([lambda x: """Rate how much you agree with the following statement about you: """ + x ],
        ["BFI base"]))

        cond_meta_instructions = list(zip([lambda x: "For the following task, respond in a way that matches this description: " + x + "\n Please respond with the single letter or number that represents your answer.",],
        ["Letter only"]))
    
        cond_coagulator = list(zip([coagulator_chat],
        ["Chat"]))
        
        with open('psytests/personas_100.txt') as f:
            cond_persona = f.readlines()[::2][:50]

    elif conditions == "personas":

        cond_options = list(zip([["A","B","C","D","E"]],
        ["ABC"]))

        cond_answers = list(zip([["Disagree strongly", "Disagree a little", "Neutral; no opinion", "Agree a little", "Agree strongly"],],
        ["Disagree Strongly to Agree Strongly"]))

        cond_p_answers = list(zip([[0,1,2,3,4]],
        ["Default"]))

        cond_p_options = list(zip([[0,1,2,3,4]],
        ["Default"]))

        cond_instructions = list(zip([lambda x: """Please indicate the extent to which you agree or disagree with the following statement: "I am someone who """ + x + """ "."""[1]],
        ["BFI base"]))

        cond_meta_instructions = list(zip([lambda x: "For the following task, respond in a way that matches this description: " + x + "\n Please respond with the single letter or number that represents your answer.",],
        ["Letter only"]))
    
        cond_coagulator = list(zip([coagulator_chat],
        ["Chat"]))
        
        with open('psytests/personas_100.txt') as f:
            cond_persona = f.readlines()[::2]

    elif conditions == "empty_personas":

        cond_options = list(zip([["A","B","C","D","E"]],
        ["ABC"]))

        cond_answers = list(zip([["Disagree strongly", "Disagree a little", "Neutral; no opinion", "Agree a little", "Agree strongly"],],
        ["Disagree Strongly to Agree Strongly"]))

        cond_p_answers = list(zip([[0,1,2,3,4]],
        ["Default"]))

        cond_p_options = list(zip([[0,1,2,3,4]],
        ["Default"]))

        cond_instructions = list(zip([lambda x: """Please indicate the extent to which you agree or disagree with the following statement: "I am someone who """ + x + """ "."""[1]],
        ["BFI base"]))

        cond_meta_instructions = list(zip([lambda x: x + "\n Please respond with the single letter or number that represents your answer.",],
        ["Letter only"]))
    
        cond_coagulator = list(zip([coagulator_chat],
        ["Chat"]))
        
        cond_persona = [""]*100

    elif conditions == "perturbations":

        cond_options = list(zip([["A","B","C","D","E"],["V","W","X","Y","Z"],["J","K","L","M","N"],["1","2","3","4","5"]],
        ["ABC","XYZ","JKL","123"]))

        cond_answers = list(zip([["Disagree strongly", "Disagree a little", "Neutral; no opinion", "Agree a little", "Agree strongly"],],
        ["Disagree Strongly to Agree Strongly"]))

        cond_p_answers = list(zip([[0,1,2,3,4],[4,3,2,1,0]],
        ["Default","Reverse"]))

        cond_p_options = list(zip([[0,1,2,3,4],[4,3,2,1,0]],
        ["Default","Reverse"]))

        cond_instructions = list(zip([lambda x: """Please indicate the extent to which you agree or disagree with the following statement: "I am someone who """ + x + """ "."""[1]],
        ["BFI base"]))

        cond_meta_instructions = list(zip([lambda x: x + "\n Please respond with the single letter or number that represents your answer.",],
        ["Letter only"]))
    
        cond_coagulator = list(zip([coagulator_chat],
        ["Chat"]))
        
        cond_persona = ["","You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. \n If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."]

    elif conditions == "perturbations_minimal":

        cond_options = list(zip([["1","2","3","4","5"]],
        ["123"]))

        cond_answers = list(zip([["Disagree", "Slightly disagree", "Neutral", "Slightly agree", "Agree"],],
        ["Disagree to Agree"]))

        cond_p_answers = list(zip([[0,1,2,3,4]],
        ["Default"]))

        cond_p_options = list(zip([[0,1,2,3,4]],
        ["Default"]))

        cond_instructions = list(zip([lambda x: """Rate how much you agree with the following statement about you: """ + x ],
        ["BFI base"]))

        cond_meta_instructions = list(zip([lambda x: x + "\n Please respond with the single letter or number that represents your answer.",],
        ["Letter only"]))
    
        cond_coagulator = list(zip([coagulator_chat],
        ["Chat"]))
        
        cond_persona = ["","You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. \n If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."]

    #All conditions are zips with items,names
    else:
        assert False, conditions

    print("Total conditions: ", len(cond_options)*len(cond_answers)*len(cond_p_answers)*len(cond_p_options)*len(cond_instructions)*len(cond_meta_instructions)*len(cond_persona)*len(cond_coagulator))
    
    return cond_options,cond_answers,cond_p_answers,cond_p_options,cond_instructions,cond_meta_instructions,cond_persona,cond_coagulator




class AnswerModel():
    def __init__(self):
        None

class SimpleNonMLAnswerModel(AnswerModel):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = torch.nn.Softmax(-1)
    def apply(self, questions: list[str], answers: list[list[str]], batch_size: int = 16):
        question_batches = np.array_split(questions, batch_size)
        answer_batches = np.array_split(answers, batch_size)

        outputs_total = []
        for i in range(len(question_batches)):
            question_batch = question_batches[i]
            answer_batch = answer_batches[i]
            outputs = self.model.get_answers(question_batch, answer_batch)
            for j in range(len(outputs)):
                outputs_total.append(self.softmax(outputs[j]))
        return outputs_total


class SimplePytorchAnswerModel(AnswerModel):
    def __init__(self, model, text_encoder):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = model.to(self.device)
        self.text_encoder = text_encoder
        self.softmax = torch.nn.Softmax(-1)
    
    def apply(self, questions: list[str], answers: list[list[str]], batch_size: int = 16, max_length: int = 2048):
        #vectorize questions and answers
        question_vectors = [self.text_encoder(question) for question in questions]
        answer_vectors =  [[self.text_encoder(a) for a in answer] for answer in answers]
        #convert to torch tensors
        question_tensors = torch.tensor(question_vectors, dtype=torch.float32).to(self.device)
        answer_tensors = torch.tensor(answer_vectors, dtype=torch.float32).to(self.device)
        
        question_batches = torch.chunk(question_tensors, batch_size)
        answer_batches = torch.chunk(answer_tensors, batch_size)
        
        outputs_total = []
        for i in range(len(question_batches)):
            question_batch = question_batches[i]
            answer_batch = answer_batches[i]
            outputs = self.model(question_batch, answer_batch)
            for j in range(len(outputs)):
                outputs_total.append(self.softmax(outputs[j].detach().cpu().numpy()))
        return outputs_total

class LLMAnswerModel(AnswerModel):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = model.to(self.device)
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.options = [" A", " B", " C", " D", " E", " F", " G", " H", " I", " J"] #Use space, as models seem to consistently "prefer" answers like that.

        assert sum([1 for i in self.tokenizer(self.options)["input_ids"] if
                    len(i) > 1]) == 0, "Cannot directly infer letter tokens"
        self.option_tokens = [i[0] for i in self.tokenizer(self.options)["input_ids"]]
        print(self.option_tokens)
        self.softmax = torch.nn.Softmax(-1)

    def apply(self, questions: list[str], answers: list[list[str]], batch_size: int = 16, max_length: int = 2048,temperature = 1.0):
        # Create formatted prompts
        prompts = [
            questions[i] + "\n" + "\n".join([self.options[j] + ": " + answers[i][j] for j in range(len(answers[i]))]) + "\n Answer:"
            for i in range(len(questions))]
        # Tokenize
        tokens = self.tokenizer(prompts, padding=True, max_length=max_length, truncation=True)
        # Note down index of predicted token
        total_tokens = [sum(mask) for mask in tokens["attention_mask"]]
        # Select actual tokens
        tokens = tokens["input_ids"]
        # Split into batches
        token_batches = [tokens[i * batch_size:(i + 1) * batch_size] for i in range(ceil(len(tokens) / batch_size))]
        totals_batches = [total_tokens[i * batch_size:(i + 1) * batch_size] for i in
                          range(ceil(len(tokens) / batch_size))]

        outputs_total = []
        for batch, totals in tqdm(zip(token_batches, totals_batches)):
            # Calculate LLM outputs
            outputs = self.model(torch.tensor(batch).to(self.device)).logits/temperature
            for i in range(len(outputs)):
                if sum(self.softmax(outputs[i, totals[i] - 1])[self.option_tokens[:len(answers[i])]]) < 0.25:
                    print("Warning, answer options have low total probability. Maybe there is an issue with the tokenizer?")
                outputs_total.append(
                    self.softmax(outputs[i, totals[i] - 1, self.option_tokens[:len(answers[i])]]).detach().cpu().numpy())


        return outputs_total

class OpenAIAnswerModel(AnswerModel):
    def __init__(self, model_name,key,chat_mode=False, argmaxing=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model_name = model_name
        with open(key) as reader:
            openai.api_key = reader.read()
        if not chat_mode:
            self.options = [" A", " B", " C", " D", " E", " F", " G", " H", " I", " J"] #Openai seems to return tokens with space in front
        else:
            self.options = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        self.softmax = torch.nn.Softmax(-1)
        self.argmaxing = argmaxing
        self.chat_mode = chat_mode

    def apply(self, questions: list[str], answers: list[list[str]]):
        # Create formatted prompts
        prompts = [
            questions[i] + "\n" + "\n".join([self.options[j] + ": " + answers[i][j] for j in range(len(answers[i]))]) + "\n Answer:"
            for i in range(len(questions))]
        if not self.chat_mode:
            response = openai.Completion.create(
                model=self.model_name,
                prompt=prompts,
                temperature=1.0,
                max_tokens=1,
                logprobs=5,
            )
            logprobs = [response["choices"][i].logprobs.top_logprobs[0] for i in range(len(prompts))]
            ps_missing = [
                min(np.min(np.exp([ps[key] for key in ps.keys()])), 1 - np.sum(np.exp([ps[key] for key in ps.keys()])))
                for ps in logprobs]
            outputs_total = []
            for i in range(len(questions)):
                outputs = []
                for j in range(len(answers[i])):
                    if self.options[j] in logprobs[i].keys():
                        outputs.append(np.exp(logprobs[i][self.options[j]]))
                    else:
                        outputs.append(ps_missing[i])
                outputs_total.append(outputs / np.sum(outputs))
        else:
            outputs_total = []
            for i, prompt in enumerate(prompts):
                print(i)
                success = False
                while not success:
                    try:
                        completion = openai.ChatCompletion.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=1,
                            max_tokens=1,
                        )
                        success = True
                    except Exception as e:
                        print(e)
                        sleep(1)

                completion = completion["choices"][0]["message"]["content"]
                out = np.zeros(len(answers[i]))
                if completion in self.options[:len(answers[i])]:
                    out[self.options[:len(answers[i])].index(completion)] = 1.0
                else:
                    print("No letter response:", completion)
                    out[np.random.randint(0, len(out))] = 1.0
                outputs_total.append(out)
        if self.argmaxing:
            return temp_zero(outputs_total)
        else:
            return outputs_total

def get_sentence_embeddings(dataset,model):
    embedds = []
    for text in dataset:
        try:
            embedds.append(np.mean(np.stack([model[w] for w in text.split(" ") if w in model.key_to_index], 0), 0))
        except:
            #print("Skipped", text)
            embedds.append(np.zeros(embedds[-1].shape))
    return np.array(embedds)

def get_sentence_embeddings_single(text,model):
    embedds = []
    try:
        embedds.append(np.mean(np.stack([model[w] for w in text.split(" ") if w in model.key_to_index], 0), 0))
    except:
        #print("Skipped", text)
        embedds.append(np.zeros(embedds[-1].shape))
    return np.array(embedds)

def temp_zero(predictions):
    out = np.zeros_like(predictions)
    out[np.arange(len(predictions)),np.argmax(predictions,-1)] = 1
    return out

"""
class StupidRegressionModel(AnswerModel):
    _model = api.load('word2vec-google-news-300')

    _data = pd.concat([pd.read_json("neuroticism.jsonl", lines=True),
                      pd.read_json("agreeableness.jsonl", lines=True),
                      pd.read_json("extraversion.jsonl", lines=True),
                      pd.read_json("conscientiousness.jsonl", lines=True),
                      pd.read_json("openness.jsonl", lines=True)])

    _labels = np.array(
        [0 for i in range(1000)] + [2 for i in range(1000)] + [4 for i in range(1000)] + [6 for i in range(1000)] + [8 for i in range(1000)])
    _labels[np.array(_data["answer_matching_behavior"] == " No")] = _labels[np.array(_data["answer_matching_behavior"] == " No")] + 1

    _test_data = pd.read_csv("MPI_120_Example.csv")
    _test_labels = np.zeros(len(_test_data))
    _test_labels[(_test_data["label_ocean"] == "N") & (_test_data["key"] == -1)] = 1

    _test_labels[(_test_data["label_ocean"] == "A") & (_test_data["key"] == 1)] = 2
    _test_labels[(_test_data["label_ocean"] == "A") & (_test_data["key"] == -1)] = 3

    _test_labels[(_test_data["label_ocean"] == "E") & (_test_data["key"] == 1)] = 4
    _test_labels[(_test_data["label_ocean"] == "E") & (_test_data["key"] == -1)] = 5

    _test_labels[(_test_data["label_ocean"] == "C") & (_test_data["key"] == 1)] = 6
    _test_labels[(_test_data["label_ocean"] == "C") & (_test_data["key"] == -1)] = 7

    _test_labels[(_test_data["label_ocean"] == "O") & (_test_data["key"] == 1)] = 8
    _test_labels[(_test_data["label_ocean"] == "O") & (_test_data["key"] == -1)] = 9

    _lg = LogisticRegression()
    _lg.fit(get_sentence_embeddings(_data["statement"],_model), _labels)


    #print("test_accuracy",_lg.score(get_sentence_embeddings(_test_data["text"],_model),_test_labels))

    #print("test_confusion \n",confusion_matrix(_test_labels, _lg.predict(get_sentence_embeddings(_test_data["text"],_model))))

    def __init__(self,o=3,c=3,e=3,a=3,n=3,deterministic=True,random_weights=False):
        super().__init__()
        self.o = o
        self.c = c
        self.e = e
        self.a = a
        self.n = n
        self.deterministic=deterministic
        self.random_weights = random_weights
        if random_weights:
            self.model = deepcopy(StupidRegressionModel._lg)
            self.model.coef_ = np.random.randn(*self.model.coef_.shape)
            self.model.intercept_ = np.random.randn(*self.model.intercept_.shape)
    def apply(self, question: str, answers: list[str]):
        #print([len(options) for options in answers])
        #assert np.all([len(options)==5 for options in answers])
        # questions = [question.split("\n")[0].split("You ")[-1] for question in questions]
        embeddings = get_sentence_embeddings_single(question,StupidRegressionModel._model)
        if not self.random_weights:
            predictions = StupidRegressionModel._lg.predict_proba(embeddings)
        else:
            predictions = self.model.predict_proba(embeddings)

        if self.deterministic:
            predictions = temp_zero(predictions)

        outputs = np.round(np.sum(predictions*np.array([[5-self.n, self.n -1,5-self.a, self.a -1,5-self.e, self.e -1,5-self.c, self.c -1,5-self.o, self.o -1,]]),-1)).astype(int)

        one_hots = np.zeros((len(outputs),5))
        one_hots[np.arange(len(outputs)),outputs] = 1

        #format should be one-hot...
        return one_hots, predictions

"""

# class StupidMultiRegressionModel(AnswerModel):
#     _model = api.load('word2vec-google-news-300')

#     _data = pd.concat([pd.read_json("neuroticism.jsonl", lines=True),
#                       pd.read_json("agreeableness.jsonl", lines=True),
#                       pd.read_json("extraversion.jsonl", lines=True),
#                       pd.read_json("conscientiousness.jsonl", lines=True),
#                       pd.read_json("openness.jsonl", lines=True)])

#     _labels = np.array(
#         [0 for i in range(1000)] + [2 for i in range(1000)] + [4 for i in range(1000)] + [6 for i in range(1000)] + [8 for i in range(1000)])
#     _labels[np.array(_data["answer_matching_behavior"] == " No")] = _labels[np.array(_data["answer_matching_behavior"] == " No")] + 1

#     _lg_n = LogisticRegression()
#     _lg_n.fit(get_sentence_embeddings(_data["statement"],_model)[_labels<2], _labels[_labels<2])

#     _lg_a = LogisticRegression()
#     _lg_a.fit(get_sentence_embeddings(_data["statement"],_model)[(_labels<4)*(_labels>1)], _labels[(_labels<4)*(_labels>1)])

#     _lg_e = LogisticRegression()
#     _lg_e.fit(get_sentence_embeddings(_data["statement"],_model)[(_labels<6)*(_labels>3)], _labels[(_labels<6)*(_labels>3)])

#     _lg_c = LogisticRegression()
#     _lg_c.fit(get_sentence_embeddings(_data["statement"],_model)[(_labels<8)*(_labels>5)], _labels[(_labels<8)*(_labels>5)])

#     _lg_o = LogisticRegression()
#     _lg_o.fit(get_sentence_embeddings(_data["statement"],_model)[(_labels<10)*(_labels>7)], _labels[(_labels<10)*(_labels>7)])

#     _lg_meta = LogisticRegression()
#     _lg_meta.fit(get_sentence_embeddings(_data["statement"],_model), np.floor(_labels/2).astype(int))

#     def __init__(self,o=3,c=3,e=3,a=3,n=3,deterministic=True):
#         super().__init__()
#         self.o = o
#         self.c = c
#         self.e = e
#         self.a = a
#         self.n = n
#         self.deterministic = deterministic
#     def apply(self, questions: list[str], answers: list[list[str]]):
#         assert np.all([len(options)==5 for options in answers])
#         questions = [question.split("\n")[0].split("You ")[-1] for question in questions]
#         embeddings = get_sentence_embeddings(questions,StupidMultiRegressionModel._model)

#         predictions_n = StupidMultiRegressionModel._lg_n.predict_proba(embeddings)
#         predictions_a = StupidMultiRegressionModel._lg_a.predict_proba(embeddings)
#         predictions_e = StupidMultiRegressionModel._lg_e.predict_proba(embeddings)
#         predictions_c = StupidMultiRegressionModel._lg_c.predict_proba(embeddings)
#         predictions_o = StupidMultiRegressionModel._lg_o.predict_proba(embeddings)
#         predictions_meta = StupidMultiRegressionModel._lg_meta.predict_proba(embeddings)

#         if self.deterministic:
#             predictions_meta = temp_zero(predictions_meta)
#             predictions_o = temp_zero(predictions_o)
#             predictions_c = temp_zero(predictions_c)
#             predictions_e = temp_zero(predictions_e)
#             predictions_a = temp_zero(predictions_a)
#             predictions_n = temp_zero(predictions_n)

#         implied_n = np.sum(predictions_n * np.array([[6-self.n,self.n]]),-1)
#         implied_a = np.sum(predictions_a * np.array([[6-self.a, self.a]]), -1)
#         implied_e = np.sum(predictions_e * np.array([[6-self.e,self.e]]),-1)
#         implied_c = np.sum(predictions_c * np.array([[6-self.c, self.c]]), -1)
#         implied_o = np.sum(predictions_o * np.array([[6-self.o, self.o]]), -1)

#         outputs = (np.round(np.sum(np.stack([implied_n,implied_a,implied_e,implied_c,implied_o],1)*predictions_meta,1))-1).astype(int)
#         one_hots = np.zeros((len(outputs),5))
#         one_hots[np.arange(len(outputs)),outputs] = 1

#         return one_hots


def calculate_mean_facet(df_test):
    means = []
    for i in range(0,len(df_test.index)):
        means.append(np.mean(df_test.values[i]))
    return means



class Test():
    def __init__(self, questions: list[str], answers: list[list[str]],
                 metrics: dict[str, Callable[[list[np.array]], float]]):
        self.questions = questions
        self.answers = answers
        self.metrics = metrics

def evaluate(test: Test, model: AnswerModel, **kwargs):
    outputs = model.apply(test.questions, test.answers, **kwargs)
    return {key: test.metrics[key](outputs) for key in test.metrics.keys()}

def cronbach_alpha(df):
    #copied from https://towardsdatascience.com/cronbachs-alpha-theory-and-application-in-python-d2915dd63586
    # 1. Transform the df into a correlation matrix
    df_corr = df.corr()
    
    # 2.1 Calculate N
    # The number of variables equals the number of columns in the df
    N = df.shape[1]
    
    # 2.2 Calculate R
    # For this, we'll loop through the columns and append every
    # relevant correlation to an array calles "r_s". Then, we'll
    # calculate the mean of "r_s"
    rs = np.array([])
    for i, col in enumerate(df_corr.columns):
        sum_ = df_corr[col][i+1:].values
        rs = np.append(sum_, rs)
    mean_r = np.mean(rs)
    
   # 3. Use the formula to calculate Cronbach's Alpha 
    cronbach_alpha = (N * mean_r) / (1 + (N - 1) * mean_r)
    return cronbach_alpha

class NEO120ScaleProperties():
    def __init__(self):
        #self.treat_model_as_sample = treat_model_as_sample
        self.data = pd.read_csv("NEO120.csv")
        Prefix =  """Given a statement of you: "You """
        Postfix =  """\n Please choose from the following options to identify how accurately this statement describes you."""
        self.questions = [Prefix+text+Postfix for text in list(self.data["text"])]
        self.answers = [["Very Accurate","Moderately Accurate","Neither Accurate Nor Inaccurate","Moderately Inaccurate","Very Inaccurate"] for i in range(len(self.questions))]
       
        self.data.to_csv('./test.csv', index=False)

    def create_sample(self,bag_of_models):
        sample =[]
        val_array = np.zeros(120)
        val_array = val_array + np.array(self.data["key"] == 1)
        val_array = val_array - np.array(self.data["key"] == -1)
        #Scoring goes from 5-1 for positive, 1-5 for negative. => Renormalize 3->0 and multiply by 2,1,0,-1,-2
        val_array = np.expand_dims(val_array,-1) * np.array([[2,1,0,-1,-2]]) + 3
        i = 0
        for model in bag_of_models:
            outputs = model.apply(self.questions, self.answers)
            # if i % 1000 == 0:
            #     print(outputs)
            answer_scores = np.sum(outputs*val_array, axis=-1)
            self.data[str(i)] = answer_scores
            i+=1
            #sample.append(answer_scores)
        return self.data
    
    def get_answer(self, model):
        val_array = np.zeros(120)
        val_array = val_array + np.array(self.data["key"] == 1)
        val_array = val_array - np.array(self.data["key"] == -1)
        #Scoring goes from 5-1 for positive, 1-5 for negative. => Renormalize 3->0 and multiply by 2,1,0,-1,-2
        val_array = np.expand_dims(val_array,-1) * np.array([[2,1,0,-1,-2]]) + 3

        outputs, predictions = model.apply(self.questions, self.answers)
        answer_scores = np.sum(outputs*val_array, axis=-1)
        return outputs, answer_scores, predictions
    
    # def get_raw_alpha_reliabiliteis(self):
    #     labels = self.data["label_raw"].unique().tolist()
    #     output = pd.DataFrame(columns = ["Facet","Alpha"])
    #     alphas = []
    #     for label in labels:
    #         #calculate alpha reliability between items of a raw label
            
    #         df = self.data[self.data["label_raw"] == label]
    #         answers = df["answer_scores"].values
    #         columns = [label]
    #         df_cronbach = pd.DataFrame(columns=columns, data = answers)
    #         alpha = cronbach_alpha(df_cronbach)
    #         alphas.append(alpha)
    #     output["Facet"] = labels
    #     output["Alpha"] = alphas
    #     return output


