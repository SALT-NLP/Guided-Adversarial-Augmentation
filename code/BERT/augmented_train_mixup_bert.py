
import argparse
import glob
import logging as log
import os
import random
import time
import torch.nn.functional as F

import numpy as np
import torch
from eval_utils import f1_score, precision_score, recall_score, classification_report, macro_score
from utils import gen_knn_mix_batch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import pickle
from transformers import *
from read_data import *

from tensorboardX import SummaryWriter

from bert_models import BertModel4Mix

logger = log.getLogger(__name__)

use_cuda = torch.cuda.is_available()
#CUDA_VISIBLE_DEVICES=6,7
#os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
MODEL_CLASSES = {"bert": (BertConfig, BertForTokenClassification, BertTokenizer)}

parser = argparse.ArgumentParser(description='PyTorch BaseNER')

parser.add_argument("--data-dir", default = './data', type = str, required = True)
parser.add_argument("--model-type", default = 'bert', type = str)
parser.add_argument("--model-name", default = 'bert-base-cased', type = str)
parser.add_argument("--output-dir", default = './german_eval', type = str)
parser.add_argument('--gpu', default='0,1,2,3', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--train-examples', default = -1, type = int)

parser.add_argument("--labels", default = "", type = str)
parser.add_argument('--config-name', default = '', type = str)
parser.add_argument("--tokenizer-name", default = '', type = str)
parser.add_argument("--max-seq-length", default = 128, type = int)

parser.add_argument("--do-train", action="store_true", help="Whether to run training.")
parser.add_argument("--do-eval", action="store_true", help="Whether to run eval on the dev set.")
parser.add_argument("--do-predict", action="store_true", help="Whether to run predictions on the test set.")
parser.add_argument("--evaluate-during-training", action="store_true", help="Whether to run evaluation during training at each logging step.")
parser.add_argument("--do-lower-case", action="store_true", help="Set this flag if you are using an uncased model.")

parser.add_argument("--batch-size", default = 16, type = int)
parser.add_argument('--eval-batch-size', default = 128, type = int)

parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")

parser.add_argument("--learning-rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight-decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam-epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

parser.add_argument("--num-train-epochs", default=20, type=float, help="Total number of training epochs to perform.")
parser.add_argument("--max-steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument('--warmup-steps', default = 0, type = int,  help="Linear warmup over warmup_steps.")

parser.add_argument('--logging-steps', default = 150, type = int, help="Log every X updates steps.")
parser.add_argument("--save-steps", type=int, default=0, help="Save checkpoint every X updates steps.")
parser.add_argument("--eval-all-checkpoints", action="store_true", help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
parser.add_argument("--overwrite-output-dir", action="store_true", help="Overwrite the content of the output directory")

parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

parser.add_argument("--pad-subtoken-with-real-label", action="store_true", help="give real label to the padded token instead of `-100` ")
parser.add_argument("--subtoken-label-type",default='real', type=str,help="[real|repeat|O] three ways to do pad subtoken with real label. [real] give the subtoken a real label e.g., B -> B I. [repeat] simply repeat the label e.g., B -> B B. [O] give it a O label. B -> B O")


parser.add_argument("--eval-pad-subtoken-with-first-subtoken-only", action="store_true", help="only works when --pad-subtoken-with-real-label is true, in this mode, we only test the prediction of the first subtoken of each word (if the word could be tokenized into multiple subtoken)")
parser.add_argument("--label-sep-cls", action="store_true", help="label [SEP] [CLS] with special labels, but not [PAD]") 



parser.add_argument('--mix-layers-set', nargs='+', default = [6,9,12], type=int)  
parser.add_argument('--alpha-regular', default=0.75, type=float)
parser.add_argument('--beta-regular', default=-1, type=float)
parser.add_argument('--alpha-aug', default=0.75, type=float)
parser.add_argument('--beta-aug', default=-1, type=float)



parser.add_argument("--log-file", default = "results.csv", type = str,help="the file to store resutls")


parser.add_argument("--optimizer", default = "adam", type = str,help='optimizer')
parser.add_argument('--special-label-weight', default=0, type=float, help='the special_label_weight in training . default 0')
parser.add_argument("--augmented-train-percentage", default = 5, type = int)






args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
args.n_gpu = torch.cuda.device_count()
print("gpu num: ", args.n_gpu)

best_f1 = 0

#print('perform mix: ', args.mix_option)
print("mix layers sets: ", args.mix_layers_set)

    

def set_seed(args):
    logger.info("random seed %s", args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if len(args.gpu) > 0:
        torch.cuda.manual_seed_all(args.seed)

def gen_aug_mix_batch(augalpha,augbeta,regalpha,regbeta,minbool,example_to_aug_example,aug_example_to_example_dict,batch,train_dataset,train_size):

    #1. fatch the sent_id from batch
    sent_id_batch = batch[5]
    
    #2. sample a sent_id for each sent 
    B_mix_batch=[]
    A_mix_batch=[]
    regular_nomix_batch=[]
    
    l=0
    l_list=[]
   
    if (200-0)<1e-6:
        l_reg=1
    else:
        if regbeta==-1:
            l = np.random.beta(regalpha, regalpha)
        else:
            l = np.random.beta(regalpha, regbeta)
        l_reg = max(l, 1-l)
        

    if (augalpha-0)<1e-6:
        l_aug=1
    else:
        if augbeta==-1:
            l = np.random.beta(augalpha, augalpha)
        else:
            l = np.random.beta(augalpha, augbeta)
        
        l_aug = max(l, 1-l)


    for sent_id in sent_id_batch:
               
        sentenceid=sent_id.cpu().numpy()
        s=int(sentenceid)
        if s in example_to_aug_example.keys():
            A_mix_batch.append(s)           
            B_mix_batch.append(example_to_aug_example[s])


            
            
            l_list.append(l_reg)

        elif s in aug_example_to_example_dict.keys():
            A_mix_batch.append(s)
            B_mix_batch.append(aug_example_to_example_dict[s])

            
            l_list.append(l_aug)
        

           
       
        else:
            regular_nomix_batch.append(s)
      
    #3. make the batch 
    B_mix_batch_datapoints = train_dataset[B_mix_batch]
    A_mix_batch_datapoints = train_dataset[A_mix_batch]
    reg_nomix_batch = train_dataset[regular_nomix_batch]

    l_tensor=torch.ones(len(A_mix_batch)).cuda()
    j=0
    for i in l_list:
        l_tensor[j]=i
        j=j+1

    l_expanded=l_tensor[:, None,None]
    return B_mix_batch_datapoints,A_mix_batch_datapoints,reg_nomix_batch,l_expanded

def read_data_rule_based_aug(args, tokenizer, labels, pad_token_label_id, mode, 
              omit_sep_cls_token=False,
              pad_subtoken_with_real_label=False):
    logger.info("Creating features from dataset file at %s", args.data_dir)
    examples = read_examples_from_file("data/conll2003",mode) 
    augexamples=[]

    if args.augmented_train_percentage>0:
        file_name="augexamples"+str(args.augmented_train_percentage)+"percent.pkl"
    else:
        augmented_train_percentage=-1*args.augmented_train_percentage
        file_name="zeroshotaugexamples"+str(augmented_train_percentage)+"percent_no_held_out_phrases.pkl"
    file_path = os.path.join(args.data_dir, file_name)
    open_file = open(file_path, "rb")
    augexamples = pickle.load(open_file)
    open_file.close()

    logger.info("Number of Augmented Examples: %s", len(augexamples))
    
    if mode =='train':
        
       
        example_to_aug_example_dict={}
        aug_example_to_example_dict={}

                     
        for augexample in augexamples:
            sent_id = int(augexample.guid.split('-')[1])
            augsentid=len(examples)
            augexample.guid="train-"+str(augsentid)
    
            if sent_id not in example_to_aug_example_dict.keys():
        
        
                example_to_aug_example_dict[sent_id]=augsentid
                aug_example_to_example_dict[augsentid]=sent_id                   
                examples.append(augexample)
             
        print("Aug Examples length ",len(aug_example_to_example_dict.keys()))
        logger.info("Aug Examples length: %s", len(aug_example_to_example_dict.keys()))

    elif mode =='test':
        examples = read_examples_from_file_excel("data/conll2003",mode)

    if mode is  'train':
        #examples = examples[0]
        print(mode)
        print('data num: {}'.format(len(examples)))

        features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer, 
                                    cls_token = tokenizer.cls_token, sep_token =  tokenizer.sep_token, pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                    cls_token_segment_id = 2 if args.model_type in ["xlnet"] else 0, 
                                    sequence_a_segment_id = 0, pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0, 
                                    pad_token_label_id = pad_token_label_id,
                                    omit_sep_cls_token=omit_sep_cls_token,
                                    pad_subtoken_with_real_label=pad_subtoken_with_real_label,
                                    subtoken_label_type=args.subtoken_label_type,
                                    label_sep_cls=args.label_sep_cls)
        
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_subtoken_ids = torch.tensor([f.subtoken_ids for f in features], dtype=torch.long)
        all_sent_id = torch.tensor([f.sent_id for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_subtoken_ids,all_sent_id)
        
      
        
        return dataset,example_to_aug_example_dict, aug_example_to_example_dict
    
    if  mode is not 'train':
        #examples = examples[0]
        print(mode)
        print('data num: {}'.format(len(examples)))

        features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer, 
                                    cls_token = tokenizer.cls_token, sep_token =  tokenizer.sep_token, pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                    cls_token_segment_id = 2 if args.model_type in ["xlnet"] else 0, 
                                    sequence_a_segment_id = 0, pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0, 
                                    pad_token_label_id = pad_token_label_id,
                                    omit_sep_cls_token=omit_sep_cls_token,
                                    pad_subtoken_with_real_label=pad_subtoken_with_real_label,
                                    subtoken_label_type=args.subtoken_label_type,
                                    label_sep_cls=args.label_sep_cls)
        
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_subtoken_ids = torch.tensor([f.subtoken_ids for f in features], dtype=torch.long)
        all_sent_id = torch.tensor([f.sent_id for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_subtoken_ids,all_sent_id)
        
        return dataset
    
def linear_rampup(current, rampup_length=args.num_train_epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

def train(args,train_dataset, eval_dataset, test_dataset_regular,test_dataset_challenging, model, tokenizer, labels, pad_token_label_id,example_to_aug_example,aug_example_to_example_dict,unlabeled_dataset=None):#example_to_aug_example second to last param
    
    global best_f1
    tb_writer = SummaryWriter()
    print('tb_writer.logdir',tb_writer.logdir)
    
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    labeled_dataloader = train_dataloader
    
    augalpha=args.alpha_aug
    augbeta=args.beta_aug
    regalpha= args.alpha_regular
    regbeta=args.beta_regular
    logger.info("regalpha: %s", regalpha)
    logger.info("regbeta: %s", regbeta)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]


    if args.n_gpu > 1:

        model = torch.nn.DataParallel(model)
    if args.optimizer=='adam':
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optimizer=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Total train batch size (w. parallel, accumulation) = %d",
        args.batch_size
        * args.gradient_accumulation_steps),
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    
    
    #eval_f1 = []
    test_f1 = []
    test_f1_regular = []
    test_f1_challenging = []
    model.zero_grad()
    minbool=True
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc='Epoch')
    set_seed(args)



    for epoch in train_iterator:

       

        epoch_iterator = tqdm(train_dataloader, desc="Iteration")        
        for step, batchorig in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            
                
            batchorig = tuple(t.to(args.device) for t in batchorig)
            #inputs_a = {"input_ids": batch[0],"attention_mask": batch[1],'subtoken_ids':batch[4]}
            #target_a=batch[3]
            # set inputs A and inputs B

       
            batch_b,batch,batch_noaugpair,l = gen_aug_mix_batch(augalpha,augbeta,regalpha,regbeta,minbool,example_to_aug_example,aug_example_to_example_dict,batch=batchorig,train_dataset=train_dataset,train_size=args.train_examples)
            inputs_a = {"input_ids": batch[0].to(args.device),"attention_mask": batch[1].to(args.device),'subtoken_ids':batch[4].to(args.device)}
            target_a=batch[3].to(args.device)    

            inputs_c = {"input_ids": batch_noaugpair[0].to(args.device),"attention_mask": batch_noaugpair[1].to(args.device),'subtoken_ids':batch_noaugpair[4].to(args.device)}
            target_c=batch_noaugpair[3].to(args.device)                         
            assert len(batch_b)==len(batch)
            inputs_b = {"input_ids": batch_b[0],"attention_mask": batch_b[1]}
            target_b=batch_b[3] 
            #else:
            #    idx=torch.randperm(batch[0].size(0))
            #    inputs_b = {"input_ids": batch[0][idx],"attention_mask": batch[1][idx]}
            #    target_b=batch[3][idx]
            
                
            mix_layer = np.random.choice(args.mix_layers_set, 1)[0]
            mix_layer = mix_layer -1 
                        
        
            #Mix Aug          
            
            inputs_b['input_ids'] = inputs_b['input_ids'].to(args.device)
            inputs_b["attention_mask"] = inputs_b["attention_mask"].to(args.device)
            target_b = target_b.to(args.device)  
            # mix the attention mask to be the longer one. 
            attention_mask = ((inputs_a["attention_mask"]+inputs_b["attention_mask"])>0).type(torch.long)
            attention_mask = attention_mask.to(args.device)
            outputs,loss_mix = model(inputs_a['input_ids'],target_a,inputs_b['input_ids'],
                                target_b,l, mix_layer,
                                attention_mask = attention_mask,
                                special_label_weight=args.special_label_weight,
                                subtoken_ids=None,
                                do_intra_mix=False)
    
                                    
        
            # No mix
            
            outputs_nomix,loss_nomix = model(inputs_c['input_ids'],target_c,
                                    attention_mask = inputs_c["attention_mask"],
                                    special_label_weight=args.special_label_weight,
                                    subtoken_ids=None,
                                    do_intra_mix=False)


            loss=torch.cat((loss_mix, loss_nomix), 0)
         




            if args.n_gpu >= 1:
                loss = loss.mean()
          
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
        
            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                
                


                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
         
                    print("augalpha ",augalpha)
                    print("regalpha ",regalpha)
                    # Log metrics
                    if (args.evaluate_during_training):
                        
                        results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, eval_dataset, parallel = False, mode="dev", prefix = str(global_step))
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
               

                        logger.info("Model name: %s", args.output_dir)
                        logger.info("Epoch is %s", epoch)
                        if results['f1'] >= best_f1:
                            best_f1 = results['f1']
                            results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, test_dataset_regular, parallel = False, mode="test", prefix = str(global_step))
                            test_f1_regular.append(results['f1'])
                            results, _ = evaluate(args,model, tokenizer, labels, pad_token_label_id, test_dataset_challenging, parallel = False, mode="test", prefix = str(global_step))
                            test_f1_challenging.append(results['f1'])
                            

                            output_dir = os.path.join(args.output_dir, "best")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            logger.info("Saving best model to %s", output_dir)
                            logger.info("Epochs trained is %s", epochs_trained)
                            logger.info("Epoch is %s", epoch)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model)  
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)

                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)
                    
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
             
                    logger.info("logging train info!!!")
                    logger.info("*")
              

            
        # eval and save the best model based on dev set after each epoch
        if (args.evaluate_during_training):
            
            results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, eval_dataset, parallel = False, mode="dev", prefix = str(global_step))
            for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value, global_step)

            
            if results['f1'] >= best_f1:
                best_f1 = results['f1']
                results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, test_dataset_regular, parallel = False, mode="test", prefix = str(global_step))
                test_f1_regular.append(results['f1'])
                results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, test_dataset_challenging, parallel = False, mode="test", prefix = str(global_step))
                test_f1_challenging.append(results['f1'])
                
                output_dir = os.path.join(args.output_dir, "best")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                logger.info("Saving best model to %s", output_dir)
                logger.info("Epochs trained is %s", epochs_trained)
                logger.info("Epoch is %s", epoch)
                model_to_save = (model.module if hasattr(model, "module") else model)  
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
        logger.info("Epoch is %s", epoch)
    args.tb_writer_logdir=tb_writer.logdir
    tb_writer.close()
    return global_step, tr_loss / global_step, test_f1_regular ,test_f1_challenging

def output_eval_results(out_label_list,preds_list,input_id_list,file_name):
    with open(file_name,'w') as fout:
        for i in range(len(out_label_list)):
            label=out_label_list[i]
            pred=preds_list[i]
            tokens=input_id_list[i]
            for j in range(len(label)):
                if tokens[j]=='[PAD]':
                    continue
                fout.write('{}\t{}\t{}\n'.format(tokens[j] ,label[j],pred[j]))
            fout.write('\n')


def evaluate(args, model, tokenizer, labels, pad_token_label_id,  eval_dataset = None, parallel = False, mode = 'dev', prefix = ''):


    eval_dataloader = DataLoader(eval_dataset, batch_size = args.eval_batch_size, shuffle = False)

    if parallel:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", mode + '-' + prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    all_subtoken_ids=None
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3],'sent_id' : batch[5]}
            inputs["token_type_ids"] = batch[2]
            target=inputs['labels']
            
            logits,tmp_eval_loss  = model(inputs['input_ids'],target,attention_mask = inputs["attention_mask"],
                                     special_label_weight=args.special_label_weight)
            
            

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  

            #eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            all_subtoken_ids=batch[4].detach().cpu().numpy()
            sent_id=inputs['sent_id'].detach().cpu().numpy()
            input_ids=inputs['input_ids'].detach().cpu().numpy()
        else:
       
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            all_subtoken_ids = np.append(all_subtoken_ids, batch[4].detach().cpu().numpy(), axis=0)
            sent_id = np.append(sent_id, inputs['sent_id'].detach().cpu().numpy(), axis=0)
            input_ids= np.append(input_ids, inputs['input_ids'].detach().cpu().numpy(), axis=0)

    #eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    input_id_list = [[] for _ in range(input_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if args.pad_subtoken_with_real_label  or args.label_sep_cls:

                if args.eval_pad_subtoken_with_first_subtoken_only:
                    if all_subtoken_ids[i,j] ==1: 
                        out_label_list[i].append(label_map[out_label_ids[i][j]])
                        preds_list[i].append(label_map[preds[i][j]])
                        tid=input_ids[i][j]
                        input_id_list[i].append(tokenizer.convert_ids_to_tokens([tid])[0])


                else:
                    if all_subtoken_ids[i,j] in [0,1] and out_label_ids[i, j] != pad_token_label_id:# in this case, we consider all the tokens.
                        out_label_list[i].append(label_map[out_label_ids[i][j]])
                        preds_list[i].append(label_map[preds[i][j]])            
                        input_id_list[i].append(tokenizer.convert_ids_to_tokens([input_ids[i][j]]))    
            else:
                if all_subtoken_ids[i,j] in [0,1] and out_label_ids[i, j] != pad_token_label_id:# in this case, we consider all the tokens.
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])                
                    input_id_list[i].append(tokenizer.convert_ids_to_tokens([input_ids[i][j]]))
    file_name=os.path.join(args.output_dir,'{}_pred_results.tsv'.format(mode))
    output_eval_results(out_label_list,preds_list,input_id_list,file_name)

    macro_scores=macro_score(out_label_list, preds_list)
    results = {
      
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
        'macro_f1':macro_scores['macro_f1'],
        'macro_precision':macro_scores['macro_precision'],
        'macro_recall':macro_scores['macro_recall']
    }

    logger.info("***** Eval results %s *****", mode + '-' + prefix)

    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list

def main():
    global best_f1
    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir):
        raise ValueError( "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    
    logger.setLevel(log.INFO)
    formatter = log.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    
    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
            
    fh = log.FileHandler(args.output_dir  +'/' + str(args.train_examples)+'-' + 'log.txt')
    fh.setLevel(log.INFO)
    fh.setFormatter(formatter)

    ch = log.StreamHandler()
    ch.setLevel(log.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    
    logger.info("------NEW RUN-----")

    logger.info("device: %s, n_gpu: %s", args.device, args.n_gpu)

    set_seed(args)

    labels = get_labels(args.labels)
    num_labels = len(labels)
    args.num_labels=num_labels

    pad_token_label_id = CrossEntropyLoss().ignore_index

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name,
        num_labels=num_labels,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name,
        do_lower_case=args.do_lower_case,
    )
    model_class = BertModel4Mix(config)

    model = model_class.from_pretrained(
        args.model_name,
        from_tf=bool(".ckpt" in args.model_name),
        config=config,
    )
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    
            

    if args.do_train:
       
        
        
        train_dataset,example_to_aug_example,aug_example_to_example_dict = read_data_rule_based_aug(args, tokenizer, labels, pad_token_label_id, mode = 'train', pad_subtoken_with_real_label=args.pad_subtoken_with_real_label)
       
       

        #test_dataset_challenging = read_data_rule_based_aug(args, tokenizer, labels, pad_token_label_id, mode = 'test', pad_subtoken_with_real_label=args.pad_subtoken_with_real_label)
        
        #if args.evaluate_during_training:
        eval_dataset = read_data(args, tokenizer, labels, pad_token_label_id, mode = 'dev',
                                    pad_subtoken_with_real_label=args.pad_subtoken_with_real_label)

     
        
        test_dataset_regular = read_data(args, tokenizer, labels, pad_token_label_id, mode = 'test', pad_subtoken_with_real_label=args.pad_subtoken_with_real_label)
    
        test_dataset_challenging = read_data_rule_based_aug(args, tokenizer, labels, pad_token_label_id, mode = 'test', pad_subtoken_with_real_label=args.pad_subtoken_with_real_label)

           
            
        #else:
        #    eval_dataset = None
        #    test_dataset = None
        
        global_step, tr_loss, test_f1_regular,test_f1_challenging = train(args,train_dataset, eval_dataset, test_dataset_regular, test_dataset_challenging,model, tokenizer, labels, pad_token_label_id,example_to_aug_example,aug_example_to_example_dict)#example_to_aug_example, param
        
        print("test_f1_regular",test_f1_regular)
        logger.info("test_f1_regular", test_f1_regular)
        print("test_f1_challenging",test_f1_challenging)
        logger.info("test_f1_challenging", test_f1_challenging)
        logger.info(" global_step = %s, average loss = %s, best eval f1 = %s", global_step, tr_loss,best_f1)
        
        
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training torch.save(model, os.path.join(args.output_dir,"bertbasev1.pt"))
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        output_dir = os.path.join(args.output_dir, "best")
        model = model_class.from_pretrained(output_dir)
        model.to(args.device)
        result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id,eval_dataset,mode="dev", prefix = 'final')
        print("Dev set results: ",result)
        
    if args.do_predict:
        print("Doing Predict!!!!")
        test_dataset_regular = read_data(args, tokenizer, labels, pad_token_label_id, mode = 'test', pad_subtoken_with_real_label=args.pad_subtoken_with_real_label)
    
        test_dataset_challenging = read_data_rule_based_aug(args, tokenizer, labels, pad_token_label_id, mode = 'test', pad_subtoken_with_real_label=args.pad_subtoken_with_real_label)

        output_dir = os.path.join(args.output_dir, "best")
        tokenizer = tokenizer_class.from_pretrained(output_dir, do_lower_case=args.do_lower_case)
        

        model = model_class.from_pretrained(output_dir)
        model.to(args.device)
        result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id,test_dataset_regular, mode="test", prefix = 'final')
        print("Regular test set results: ",result)
        result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id,test_dataset_challenging, mode="test", prefix = 'final')
        print("Challenging test set results: ",result)



main()
