import pytorch_lightning as pl
import pandas as pd
import logging
from pathlib import Path
import os
import re
import numpy as np
import torch
from pytorch_lightning import loggers as pl_loggers                                                                                                                            
from torch.utils.data import DataLoader, Dataset                                                                                                                               
from dataset import KoBARTGecDataset                                                                                                                                           
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast                                                                                                 
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from metric.gleumodule import run_gleu                                                                                                                                         
from transformers import set_seed                                                                                                                                              
from pprint import pprint
import time                                                                                                                                                                    

class KoBARTConditionalGeneration(pl.LightningModule):
    def __init__(self, args, model, tokenizer, datamodules):
        super().__init__()
        self.assign_attributes(args, model, tokenizer, datamodules)

    def assign_attributes(self, args, model, tokenizer, datamodules):
        self.bos_token = '<s>'                                                                                                                                            
        self.eos_token = '</s>'                                               
        self.pad_token_id = 0                                                 
        self.epoch = 0                                                                                                                                             
        self.outputs = []                                                                                                                                            
        self.decoded_labels = []                        
        self.origs = []                                       
        self.step = 0                                         
        self.args = args
        self.max_len = self.args.max_seq_len                                      
        self.scores = {}                                                      
        self.generation_time = 0
        self.model = model
        self.tokenizer = tokenizer
        self.dm = datamodules

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr, correct_bias=False)
        data_len = len(self.dm.train_dataloader().dataset)
        num_train_steps = int(data_len * self.args.max_epochs / self.args.batch_size)
        if data_len < self.args.batch_size:
            num_train_steps = self.args.max_epochs
        print(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.args.warmup_ratio)

        print(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler,
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        self.scheduler = scheduler
        return [optimizer], [lr_scheduler]

    def forward(self, inputs):                                           
        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()                                                                                   
        return self.model(input_ids=inputs['input_ids'],                      
                attention_mask=attention_mask,                                                                                                               
                decoder_input_ids=inputs['decoder_input_ids'],                
                decoder_attention_mask=decoder_attention_mask,            
                labels=inputs['labels'], return_dict=True)                                                                                                   
                                                                                                                                                             
    def training_step(self, batch, batch_idx):                                                                                                                
        outs = self(batch)                                                                                                                                      
        loss = outs.loss                                                                                                                                     
        self.log('train_loss', loss, prog_bar=False)                                                                                                          
        self.step += 1                                                                                                                                       
        return loss                                                                                                                                          
                                                                                                                                                             
    def training_epoch_end(self, _):                                                                                                                         
        self.scores[self.epoch]['generation_time'] = self.generation_time     
        print(f"\nGeneration time: {self.generation_time}")                                                                                                        
        self.generation_time = 0                                                                                                                                     
        self.epoch += 1                                                       
                                                                                                                                                             
    def generate(self, input_ids, labels):                                                                                                                           
        self.model.eval()                                                                                                                                                 
        start = time.time()                                   
        output = self.model.generate(input_ids, eos_token_id=1, max_length=self.max_len, num_beams=4)                                                                        
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)                                                                                            
        end = time.time()                                                     
        self.generation_time += end - start 
        decoded_label = self.tokenizer.batch_decode(labels.masked_fill(labels == -100, 1), skip_special_tokens=True)                                                      
        self.outputs += [x.replace('\n', '') for x in output]                 
        self.decoded_labels += decoded_label                  
        self.origs += self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)                                                                                       
                                                                              
    def should_generate(self):                                                 
        return True                                                           
                                                                              
    def validation_step(self, batch, batch_idx):              
        outs = self(batch)                                    
        loss = outs['loss']                    
        if self.should_generate():                                            
            self.generate(batch['input_ids'], batch['labels'])                
        return (loss)                                                                                                                                        
                                                                                                                                                              
    def test_step(self, batch, batch_idx):                                                                                                                      
        outs = self(batch)                                                                                                                                   
        loss = outs['loss']                                                                                                                                   
        if self.should_generate():                                                                                                                           
            self.generate(batch['input_ids'], batch['labels'])                                                                                                
        return (loss)                                                                                                                                        
    def test_epoch_end(self, outputs):                                                                                                                        
        return self.validation_epoch_end(outputs, mode='test')                                                                                                  
                                                                                                                                                                   
    def validation_epoch_end(self, outputs, mode='val'):                                                                                                             
        losses = []                                                                                                                                                                                                                                                                                                       
        for loss in outputs:
            losses.append(loss)
        total_loss = torch.stack(losses).mean()                                                                                                              
        if self.should_generate():                                                                                                                            
            # Make generation output directory and file                                                                                                         
            directory = f"outputs/generation/epoch{self.epoch}/{mode}"                                                                                                                                                                                                                               
            path = Path(directory)                                                                                                                                                                                                                                                                                          
            path.mkdir(parents=True, exist_ok=True)                                                                                                                                                                                                                                                                             
            with open(directory + f"/hypothesis_{total_loss}.txt", 'w', encoding='utf-8') as f:                                                                    
                f.write("\n".join(self.outputs))                                                                                                                   
            with open(directory + f"/reference_{total_loss}.txt", "w", encoding='utf-8') as f:                                                                                                                                                                                                                            
                f.write("\n".join(self.decoded_labels))                                                                                                                                                                                                                                                                     
            with open(directory + f"/source_{total_loss}.txt", "w", encoding='utf-8') as f:                                                                                                                                                                                                                                     
                f.write("\n".join(self.origs))                                                                                                                                                                                                                                                                                            
            self.outputs = []                                                                                                                                                                                                                                                                                                                        
            self.decoded_labels = []                                     
            self.origs = []                                                                                                                                  
            gleu_out = run_gleu(reference=directory + f"/reference_{total_loss}.txt", source=directory + f"/source_{total_loss}.txt", hypothesis=directory + f"/hypothesis_{total_loss}.txt")                                                                                                                                          
            logging.info(f"\ngleu_value: {gleu_out}\n")
            with open(directory + f"/gleu_{gleu_out}.txt", "w", encoding='utf-8') as f:                                                                                      
                f.write(f"data: {self.args.data}, epoch: {self.epoch}, gleu_out: {gleu_out}, val_loss: {total_loss}\nhparams:{self.hparams}")                                                                                                                                                                                                        
            # make m2 file
            if not os.path.exists('get_data/{self.args.data}_{mode}.m2'):
                print("Making m2 file since we don't have one..")
                command = f"cd KAGAS/ && python3 parallel_to_m2_korean.py -orig ../{directory}/source_{total_loss}.txt -cor ../{directory}/reference_{total_loss}.txt -out get_data/{self.args.data}_{mode}.m2 -noprint && cd ../"
            os.system(command)
            # output m2scores                                                                                                                                
            system_command = f"python3 ./metric/m2scorer/scripts/m2scorer.py {directory}/hypothesis_{total_loss}.txt get_data/{self.args.data}_{mode}.m2 > {directory}/m2score.txt"
            print(f"Processing: {system_command}")                                                                                                              
            #out = os.system(system_command)                                                                                                                                                                                                                                                                                                          
            #os.system(f"cat {directory}/m2score.txt")                                                                                                         
            #with open(f"{directory}/m2score.txt", encoding='utf-8') as f:                                                                                       
           #     try:                                                          
           #         p, r, f_score = [x.split(": ")[1].strip() for x in f.read().strip().split("\n")]                                                                         
            #    except:                                                         
            #        print("Reading m2 scores failed")                                                                                                                     
             #       p, r, f_score = 0, 0, 0                                                                                                                        
        p, r, f_score = 0,0,0
        gleuscore = float(gleu_out) * 100                                                                                                                            
        self.scores[self.epoch] = {'precision': p, 'recall': r, 'f_score': f_score, 'gleu': gleuscore, 'loss': total_loss.item()}                                            
        print(f"\n\nEPOCH {self.epoch} / VAL_LOSS {round(total_loss.item(), 2)} / GLEU {round(gleuscore, 2)}\n\n")                                                           
        self.log(f'{mode}_loss', total_loss, prog_bar=False)                      
        self.log(f'{mode}_gleu', gleuscore)                                          
        if self.args.best['gleu'] < gleuscore:                                       
            self.args.best['gleu'] = gleuscore                                       
            self.args.best['f0.5'] = f_score                                                                                                                              
            self.args.best['prec'] = p                                               
            self.args.best['rec'] = r                                                 
        print(f"Print ordering of: precision, recall, f0.5 score, gleu score, val loss, generation_time. The last epoch is test epoch.")                                     
        pprint(self.scores)
