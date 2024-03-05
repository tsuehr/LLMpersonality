from Utils import initialize_models,get_tests,get_conditions
import argparse
from transformers import AutoTokenizer
import torch
softmax = torch.nn.Softmax(-1)
import pandas as pd
import numpy as np
import openai
from time import sleep

try:
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-30b")
except:
    tokenizer = None

def get_prompt(instruction,item,answers,options,meta_instruction,persona):
    question = instruction(item)
    option_text = "\n".join([options[j] + ": " + answers[j] for j in range(len(answers))])
    
    main_text = question + "\n" + option_text  + "\nAnswer:" 

    meta=meta_instruction(persona) 

    return meta,main_text

def get_incontext_prompt(item, answers, options):
    option_text = "\n".join([options[j] + ": " + answers[j] for j in range(len(answers))])
    main_text = item + "\n" + option_text  + "\nAnswer:"
    return main_text

def eval_model(model=None,instruction=None,item=None,answers=None,options=None,meta_instruction = None,p_answers=None,p_options =None,persona =None,coagulator=None ,i_negated=None, context_history=None):

    options = np.array(options)
    answers = np.array(answers)
  
    #Permute answers, and associate each answer to an option
    option_map = {options[i]: answers[p_answers][i] for i in range(len(options))}
    #Permute options, create prompts for the options and associated answers
    options_in = options[p_options]
    answers_in = [option_map[option] for option in options_in]
    if context_history == "":
        meta,main_text = get_prompt(instruction,item,answers_in,options_in,meta_instruction,persona)
        ps,prompt,log,context_history = query(model,meta,main_text,options,coagulator)
    else:
        main_text = get_incontext_prompt(item,answers,options)
        ps,prompt,log,context_history = query_incontext(model,main_text,options,coagulator, context_history)

    

    #Option i is associated to answer_permutation(answers)[i] 
    #=> Apply reverse permutation to obtain answers in correct order. Argsort reverses permutations.
    ps = ps[np.argsort(p_answers)]
    #If item is negated: Reverse the scale!
    if i_negated:
        ps = ps[np.arange(len(options))[::-1]]
    return ps,prompt,log, context_history

def query_incontext(model,main_text,options,coagulator, context_history):
    if type(model) is not str:
        model_type = "Local"
    elif model in ["gpt-4-0613","gpt-4-0314","gpt-3.5-turbo-0613","gpt-3.5-turbo-0301"]:
        model_type = "OpenAI_Chat"
    elif model in ["text-davinci-003","text-davinci-002","text-davinci-001","davinci"]:
        model_type = "OpenAI_Complete"
    else:
        assert False, model
    
    if model_type == "Local":
        # prompt = coagulator(meta,main_text,model)
        prompt = context_history + "\n" + "[INST]" + main_text + "[/INST]"
        tokens = tokenizer(prompt,return_tensors="pt")["input_ids"]

        token_lengths = [len(i) for i in tokenizer(list(options))["input_ids"]]
        assert (sum(token_lengths)/len(token_lengths)) == token_lengths[0] #Either all options are numeric or all are letters
        if token_lengths[0] == 3:
            tokens = torch.cat([tokens,torch.tensor([[29871]])],1) #Append "is a number" token to the prompt
        #Evaluate model 
        with torch.no_grad():
            outputs = model(tokens.to(torch.device("cuda"))).logits[0,-1]
        #Write down tokens representing each of the options
        option_tokens = np.array([i[-1] for i in tokenizer(list(options))["input_ids"]])
        #Probabilities for the options in the original order. No need to undo the option permutation here. 
        #LLama seems to tokenize single letters without an additional space
        ps = softmax(outputs)[option_tokens].detach().cpu().numpy()
        indices,values = torch.topk(outputs,25)
        # vs = values.detach().cpu().numpy()
        # top_choice = tokenizer.decode([vs[0]])
        # if not (top_choice in options) or top_choice == "":
        #     top_choice = tokenizer.decode([vs[1]])    
        # log = "i: " + str(indices.detach().cpu().numpy()) + " v:" + str(vs)
        top_choice = options[np.argmax(ps)]
        log = top_choice
        #print(log)
        context_history = prompt + "\n" + log

    elif model_type == "OpenAI_Chat":
        #prompt = coagulator(meta,main_text,model) 
        prompt = context_history + [{"role":"user","content":main_text}]
        #Query model until it actually responds. 
        retry = True
        while retry:
            try:
                response = openai.ChatCompletion.create(model=model,messages=prompt,temperature = 1.0,max_tokens = 1)
                retry  = False
                if model in ["gpt-4-0613","gpt-4-0314"]:
                    sleep(0.3) #Respect rate limit
                else:
                    sleep(0.3) #Respect rate limit
            except Exception as e:
                print(e)
                retry = True
        
        #Write down tokens representing each of the options
        outputs = []
        for j in range(len(options)):
            #Chat models appear to tokenize single letters without an added space
            if options[j] == response.choices[0].message.content:
                outputs.append(1.0)
            else:
                outputs.append(0.0)
        
        #In case of refusal, record that separately and send notification
        ps = np.array(outputs)
        if sum(ps) == 0:
            print(prompt,"refusal")
            ps = np.ones_like(ps)/len(options)
        log = str(response.choices[0].message.content)

    elif model_type ==  "OpenAI_Complete":
        # prompt = coagulator(meta,main_text,model)
        prompt = context_history + [{"role":"user","content":main_text}]
        retry = True
        while retry:
            try:
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=1,
                    logprobs=5,
                )
                retry  = False
                sleep(0.025) #Respect rate limit
            except Exception as e:
                print(e)
                retry = True
        
        logprobs = response["choices"][0].logprobs.top_logprobs[0] 
        #Calculate imputation values for missing probabilities
        min_p = np.min(np.exp([logprobs[key] for key in logprobs.keys()]))
        missing_p =  1 - np.sum(np.exp([logprobs[key] for key in logprobs.keys()]))
        p_impute = min(min_p,missing_p)
        
        #Probabilities for the options in the original order 
        outputs = []
        for j in range(len(options)):
            #Non-chat models appear to tokenize single letters with a space in front
            if " "+options[j] in logprobs.keys():
                outputs.append(np.exp(logprobs[" "+options[j]]))
            else:
                #Impute missing probabilities
                outputs.append(p_impute)

        #Normalize probabilities
        ps = outputs / np.sum(outputs)
        log = str(dict(response.choices[0].logprobs.top_logprobs[0]))
        context_history = prompt + [{"role":"assistant","content":log}]
    return ps,prompt,log, context_history

def query(model,meta,main_text,options,coagulator):
    if type(model) is not str:
        model_type = "Local"
    elif model in ["gpt-4-0613","gpt-4-0314","gpt-3.5-turbo-0613","gpt-3.5-turbo-0301"]:
        model_type = "OpenAI_Chat"
    elif model in ["text-davinci-003","text-davinci-002","text-davinci-001","davinci"]:
        model_type = "OpenAI_Complete"
    else:
        assert False, model
    
    if model_type == "Local":
        prompt = coagulator(meta,main_text,model)
        tokens = tokenizer(prompt,return_tensors="pt")["input_ids"]

        token_lengths = [len(i) for i in tokenizer(list(options))["input_ids"]]
        assert (sum(token_lengths)/len(token_lengths)) == token_lengths[0] #Either all options are numeric or all are letters
        if token_lengths[0] == 3:
            tokens = torch.cat([tokens,torch.tensor([[29871]])],1) #Append "is a number" token to the prompt
        #Evaluate model 
        with torch.no_grad():
            outputs = model(tokens.to(torch.device("cuda"))).logits[0,-1]
        #Write down tokens representing each of the options
        option_tokens = np.array([i[-1] for i in tokenizer(list(options))["input_ids"]])
        #Probabilities for the options in the original order. No need to undo the option permutation here. 
        #LLama seems to tokenize single letters without an additional space
        ps = softmax(outputs)[option_tokens].detach().cpu().numpy()
        indices,values = torch.topk(outputs,25)
        # vs = values.detach().cpu().numpy()
        # top_choice = tokenizer.decode([vs[0]])
        top_choice = options[np.argmax(ps)]
        # log = "i: " + str(indices.detach().cpu().numpy()) + " v:" + str(vs)
        log = top_choice
        # print(log)
        ##append answer off LLM to history
        context_history = prompt + '\n'+ log

    elif model_type == "OpenAI_Chat":
        prompt = coagulator(meta,main_text,model) 
        #Query model until it actually responds. 
        retry = True
        while retry:
            try:
                response = openai.ChatCompletion.create(model=model,messages=prompt,temperature = 1.0,max_tokens = 1)
                retry  = False
                if model in ["gpt-4-0613","gpt-4-0314"]:
                    sleep(1.0) #Respect rate limit
                else:
                    sleep(1.0) #Respect rate limit
            except Exception as e:
                print(e)
                retry = True
        
        #Write down tokens representing each of the options
        outputs = []
        for j in range(len(options)):
            #Chat models appear to tokenize single letters without an added space
            if options[j] == response.choices[0].message.content:
                outputs.append(1.0)
            else:
                outputs.append(0.0)
        
        #In case of refusal, record that separately and send notification
        ps = np.array(outputs)
        if sum(ps) == 0:
            print(prompt,"refusal")
            ps = np.ones_like(ps)/len(options)
        log = str(response.choices[0].message.content)
        context_history = prompt + [{"role":"assistant","content":log}]

    elif model_type ==  "OpenAI_Complete":
        prompt = coagulator(meta,main_text,model)
        retry = True
        while retry:
            try:
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=1,
                    logprobs=5,
                )
                retry  = False
                sleep(0.025) #Respect rate limit
            except Exception as e:
                print(e)
                retry = True
        
        logprobs = response["choices"][0].logprobs.top_logprobs[0] 
        #Calculate imputation values for missing probabilities
        min_p = np.min(np.exp([logprobs[key] for key in logprobs.keys()]))
        missing_p =  1 - np.sum(np.exp([logprobs[key] for key in logprobs.keys()]))
        p_impute = min(min_p,missing_p)
        
        #Probabilities for the options in the original order 
        outputs = []
        for j in range(len(options)):
            #Non-chat models appear to tokenize single letters with a space in front
            if " "+options[j] in logprobs.keys():
                outputs.append(np.exp(logprobs[" "+options[j]]))
            else:
                #Impute missing probabilities
                outputs.append(p_impute)

        #Normalize probabilities
        ps = outputs / np.sum(outputs)
        log = str(dict(response.choices[0].logprobs.top_logprobs[0]))
        context_history = prompt + [{"role":"assistant","content":log}]
    return ps,prompt,log, context_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name',type=str)
    parser.add_argument('conditions',type=str)
    parser.add_argument('test',type=str)
    args = parser.parse_args()  
    if args.model_name in ["llama-2-7b-chat-hf","llama-2-70b-chat","llama-2-7b-hf","llama-2-70b-hf"]: 
        models = initialize_models([args.model_name],["float32"])
        model = models[args.model_name+"_float32"]
    else:
        model = args.model_name
        with open('../test.txt') as reader:
            openai.api_key = reader.read()

    cond_options,cond_answers,cond_p_answers,cond_p_options,cond_instructions,cond_meta_instructions,cond_persona,cond_coagulator = get_conditions(args.conditions)
    #All conditions except for personas are list(zip) with items,names

    tests = get_tests(args.test)
    aspect_names = ["O","E","C","A","N"]
    aspects= {test_name: {aspect_name:
            [list(tests[test_name][(tests[test_name]["label_ocean"]==aspect_name) 
                                    & (tests[test_name]["key"]==key)]["text"]) 
            for key in [1,-1]] 
            for aspect_name in aspect_names} for test_name in list(tests.keys())}
   
    result_list = []
    counter = 0
    for options,options_name in cond_options:
        for answers,answers_name in cond_answers:
            for inst,inst_name in cond_instructions:
                for meta,meta_name in cond_meta_instructions:
                    for coagulator,coagulator_name in cond_coagulator:
                        for p_answer,p_answer_name in cond_p_answers:
                            for p_option,p_option_name in cond_p_options:
                                for persona in cond_persona:
                                    counter = counter+1
                                    print(str(counter) + "/" + str(len(cond_persona)))
                                    for test_name in list(tests.keys()):
                                        context_history = ""
                                        for aspect_name in aspect_names:
                                            print(aspect_name)
                                            for reverse_key in [0,1]:
                                                for item in aspects[test_name][aspect_name][reverse_key]:

                                                    ps,prompt,log,context_history = eval_model(model=model,
                                                    instruction=inst, 
                                                    item=item,
                                                    answers=answers,
                                                    options=options,
                                                    meta_instruction = meta,
                                                    p_answers=p_answer,
                                                    p_options = p_option,
                                                    persona = persona,
                                                    coagulator=coagulator,
                                                    context_history = context_history,
                                                    i_negated = bool(reverse_key))
                                                    #print(ps,prompt,log)
                                                    result_list.append(
                                                    {"persona":persona,"aspect":aspect_name,"reverse": reverse_key,"item":item,
                                                    "options":options_name,"answers":answers_name,"reverse_answers":p_answer_name,
                                                    "reverse_options":p_option_name,"instruction":inst_name,"meta":meta_name,"coagulator":coagulator_name,
                                                    "item":item,"test_name":test_name, "p1":ps[0],"p2":ps[1],"p3":ps[2],"p4":ps[3],"p5":ps[4],
                                                    "prompt":prompt,"log":log})
                                    pd.DataFrame(result_list).to_csv("output_raw/incontext/"+args.model_name+"_"+args.conditions+"_"+args.test+".csv") #Save results once a persona-condition is fully evaluated. 
                           
    
    
    
    
    
    
