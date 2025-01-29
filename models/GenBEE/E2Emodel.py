import logging, os
import numpy as np
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoConfig, AutoModelForPreTraining,  AutoTokenizer,AutoModel
import ipdb
from .prefix_gen_bart import PrefixGenBartForConditionalGeneration
from .projector import Projector
logger = logging.getLogger(__name__)

class GenerativeModel(nn.Module):
    def __init__(self, config, tokenizer,type_set):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.type_set = type_set
        if self.config.model_type=='degree':
            self.model = DegreeE2EModel(config, tokenizer)

        elif self.config.model_type=='PreDegree':
            self.model = AMRPrefixGen(config, tokenizer)
            

        else:
            raise ValueError("Model type {} does not support yet.".format(self.config.model_type))

    def forward(self, batch):
        return self.model(batch)
        
    def predict(self, batch, num_beams=4, max_length=50):
        return self.model.predict(batch, num_beams, max_length)
    
    def generate(self, input_ids, attention_mask, context_input, num_beams=4, max_length=50, 
                **kwargs):
        self.eval()
        with torch.no_grad():
            output = self.model.generate(input_ids, attention_mask, context_input, num_beams, max_length,  **kwargs)
        self.eval()
        return output

    def save_model(self, save_path):
        """
        This save model is created mainly in case we need partial save and load. Such as cases with pretraining.
        """
        self.model.save_model(save_path)

    def load_model(self, load_path):
        """
        This load model is created mainly in case we need partial save and load. Such as cases with pretraining.
        """
        self.model.load_model(load_path)


class DegreeE2EModel(nn.Module):
    def __init__(self, config, tokenizer, type_set):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.type_set = type_set
        
        if self.config.pretrained_model_name.startswith('facebook/bart' or 'google-t5/'):
            self.model_config = AutoConfig.from_pretrained(self.config.pretrained_model_name,
                                                          cache_dir=self.config.cache_dir)
            self.model = AutoModelForPreTraining.from_pretrained(self.config.pretrained_model_name,
                                                        cache_dir=self.config.cache_dir, config=self.model_config)
        else:
            raise ValueError("Not implemented.")
            
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def process_data(self, batch):
        # encoder inputs
        inputs = self.tokenizer(batch.batch_input, return_tensors='pt', padding=True)
        enc_idxs = inputs['input_ids']
        enc_attn = inputs['attention_mask']

        # decoder inputs
        targets = self.tokenizer(batch.batch_target, return_tensors='pt', padding=True)
        batch_size = enc_idxs.size(0)
        
        if self.config.pretrained_model_name.startswith('facebook/bart'):
            padding = torch.ones((batch_size, 1), dtype=torch.long)
            padding[:] = self.tokenizer.eos_token_id
            # for BART, the decoder input should be:
            # PAD => BOS
            # BOS => A
            # A => B          
        else:
            # t5 case
            padding = torch.ones((batch_size, 1), dtype=torch.long)
            padding[:] = self.tokenizer.pad_token_id
            # for t5, the decoder input should be:
            # PAD => A
            # A => B
        
        dec_idxs = torch.cat((padding, targets['input_ids']), dim=1)
        dec_attn = torch.cat((torch.ones((batch_size, 1), dtype=torch.long), targets['attention_mask']), dim=1)
        # dec_idxs = targets['input_ids']
        # dec_idxs[:, 0] = self.tokenizer.eos_token_id
        # dec_attn = targets['attention_mask']
            
        # labels
        padding = torch.ones((batch_size, 1), dtype=torch.long)
        padding[:] = self.tokenizer.pad_token_id
        raw_lbl_idxs = torch.cat((dec_idxs[:, 1:], padding), dim=1)
        lbl_attn = torch.cat((dec_attn[:, 1:], torch.zeros((batch_size, 1), dtype=torch.long)), dim=1)
        lbl_idxs = raw_lbl_idxs.masked_fill(lbl_attn==0, -100) # ignore padding
        
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        dec_idxs = dec_idxs.cuda()
        dec_attn = dec_attn.cuda()
        raw_lbl_idxs = raw_lbl_idxs.cuda()
        lbl_idxs = lbl_idxs.cuda()
        
        return enc_idxs, enc_attn, dec_idxs, dec_attn, raw_lbl_idxs, lbl_idxs

    def forward(self, batch):
        enc_idxs, enc_attn, dec_idxs, dec_attn, raw_lbl_idxs, lbl_idxs = self.process_data(batch)
        outputs = self.model(input_ids=enc_idxs, 
                             attention_mask=enc_attn, 
                             decoder_input_ids=dec_idxs, 
                             decoder_attention_mask=dec_attn, 
                             labels=lbl_idxs, 
                             return_dict=True)
        
        loss = outputs['loss']
        
        return loss
        
    def predict(self, batch, num_beams=4, max_length=50):
        enc_idxs, enc_attn, dec_idxs, dec_attn, raw_lbl_idxs, lbl_idxs = self.process_data(batch)
        return self.generate(enc_idxs, enc_attn, num_beams, max_length)
    
    def generate(self, input_ids, attention_mask, num_beams=4, max_length=50, **kwargs):
        self.eval()
        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, 
                                          attention_mask=attention_mask, 
                                          num_beams=num_beams, 
                                          max_length=max_length)
        final_output = []
        for bid in range(len(input_ids)):
            # if self.config.model_name.startswith('google/t5') or self.config.model_name.startswith('t5'):
            #     output_sentence = t5_decode(self.tokenizer, outputs[bid])
            # else:
            #     output_sentence = self.tokenizer.decode(outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            output_sentence = self.tokenizer.decode(outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            final_output.append(output_sentence)
        self.train()
        return final_output
    
    def save_model(self, save_path):
        self.model.save_pretrained(save_path)

    def load_model(self, load_path):
        self.model.from_pretrained(load_path)



class AMRPrefixGenBase(nn.Module):
    def __init__(self, config, tokenizer):
        super(AMRPrefixGenBase, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(self.config.gpu_device)  

    def get_prefix(self, batch_type):
        global_prompt = ["Explore events that frequently co-occur with \
                            <event type> events, aiming to identify and analyze the  \
                            interactions and dependencies among these events to enhance  \
                            understanding of their interrelationships.", 
                         " Explore entities that serve as triggers in both \
                            <event type> events and other event types, aiming \
                            to clarify the overlap in trigger roles across different contexts \
                            to better understand trigger versatility.",
                         "Explore entities acting in multiple roles, including as \
                            roles in <event type> events and differently in other \
                            events, highlighting the dynamics of role versatility and their \
                            implications for event structure.", 
                         "Explore entities where the trigger of <event type> \
                            events also acts as a role in other events, or vice versa, highlighting these complex inter-event relationships to identify patterns \
                            of event interaction." \
                         ]
        modified_batch_prompt = []
        for event_type in batch_type:
            event_type_token = f"<T>{event_type}<T>"
            prompts_for_type = " ".join([prompt.replace("<event type>", event_type_token) for prompt in global_prompt])
            modified_batch_prompt.append(prompts_for_type)
        #logger.info(f"prefix_prompt: {modified_batch_prompt}")

        inputs = self.prefix_tokenizer(modified_batch_prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs  = self.roberta(**inputs)

        sentence_representation = outputs.last_hidden_state[:, 0]

        prefix_keys_flat = self.ffnn(sentence_representation)
        prefix_values_flat = prefix_keys_flat


        

        batch_size = sentence_representation.shape[0]
        prefix_keys = prefix_keys_flat.view(batch_size, self.num_layers, self.config.prefix_length, self.head_num, self.head_dim)
        prefix_values = prefix_values_flat.view(batch_size, self.num_layers, self.config.prefix_length, self.head_num, self.head_dim)
        prefix = {}
        if self.config.use_encoder_prefix:
            prefix['encoder_prefix'] = (prefix_keys, prefix_values)
        if self.config.use_cross_prefix:
            prefix['cross_prefix'] = (prefix_keys, prefix_values)
        if self.config.use_decoder_prefix:
            prefix['decoder_prefix'] = (prefix_keys, prefix_values)

        return prefix

    def forward(self, batch):
        enc_idxs, enc_attn, dec_idxs, dec_attn, raw_lbl_idxs, lbl_idxs, batch_type = self.process_data(batch)
        prefix = self.get_prefix(batch_type)
        
        outputs = self.model(input_ids=enc_idxs, 
                             prefix=prefix,
                             attention_mask=enc_attn, 
                             decoder_input_ids=dec_idxs, 
                             decoder_attention_mask=dec_attn, 
                             labels=lbl_idxs, 
                             return_dict=True)
        
        loss = outputs['loss']
        
        return loss
    

    def process_data(self, batch):
        # encoder inputs
        inputs = self.tokenizer(batch.batch_input, return_tensors='pt', padding=True)
        enc_idxs = inputs['input_ids']
        enc_attn = inputs['attention_mask']

        # decoder inputs
        targets = self.tokenizer(batch.batch_target, return_tensors='pt', padding=True)
        batch_size = enc_idxs.size(0)
        
        if self.config.pretrained_model_name.startswith('facebook/bart'):
            padding = torch.ones((batch_size, 1), dtype=torch.long)
            padding[:] = self.tokenizer.eos_token_id
            # for BART, the decoder input should be:
            # PAD => BOS
            # BOS => A
            # A => B          
        else:
            # t5 case
            padding = torch.ones((batch_size, 1), dtype=torch.long)
            padding[:] = self.tokenizer.pad_token_id
            # for t5, the decoder input should be:
            # PAD => A
            # A => B
        
        dec_idxs = torch.cat((padding, targets['input_ids']), dim=1)
        dec_attn = torch.cat((torch.ones((batch_size, 1), dtype=torch.long), targets['attention_mask']), dim=1)
        # dec_idxs = targets['input_ids']
        # dec_idxs[:, 0] = self.tokenizer.eos_token_id
        # dec_attn = targets['attention_mask']
            
        # labels
        padding = torch.ones((batch_size, 1), dtype=torch.long)
        padding[:] = self.tokenizer.pad_token_id
        raw_lbl_idxs = torch.cat((dec_idxs[:, 1:], padding), dim=1)
        lbl_attn = torch.cat((dec_attn[:, 1:], torch.zeros((batch_size, 1), dtype=torch.long)), dim=1)
        lbl_idxs = raw_lbl_idxs.masked_fill(lbl_attn==0, -100) # ignore padding
        batch_type = batch.batch_event_type

        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        dec_idxs = dec_idxs.cuda()
        dec_attn = dec_attn.cuda()
        raw_lbl_idxs = raw_lbl_idxs.cuda()
        lbl_idxs = lbl_idxs.cuda()
        
        return enc_idxs, enc_attn, dec_idxs, dec_attn, raw_lbl_idxs, lbl_idxs, batch_type

    

    def predict(self, batch, num_beams=4, max_length=50):
        enc_idxs, enc_attn, dec_idxs, dec_attn, raw_lbl_idxs, lbl_idxs, batch_type = self.process_data(batch)
        return self.generate(enc_idxs, enc_attn, batch_type, num_beams, max_length)
    
    def generate(self, input_ids, attention_mask, batch_type, num_beams=4, max_length=50, **kwargs):
        self.eval()
        with torch.no_grad():
            batch_prefix = self.get_prefix(batch_type)
            self.model._cache_input_ids = input_ids
            prefix = batch_prefix
                

                
            model_kwargs = {'prefix': prefix}
            outputs = self.model.generate(input_ids=input_ids, 
                                          attention_mask=attention_mask, 
                                          num_beams=num_beams, 
                                          max_length=max_length,
                                          **model_kwargs)


        final_output = []
        for bid in range(len(input_ids)):
            output_sentence = self.tokenizer.decode(outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            final_output.append(output_sentence)
        self.train()
        return final_output
    
    
    def save_model(self, save_path):
        self.model.save_pretrained(os.path.join(save_path, "checkpoint-bart"))
        torch.save(self.roberta.state_dict(), os.path.join(save_path, "roberta.mdl"))
        torch.save(self.ffnn.state_dict(), os.path.join(save_path, "ffnn.mdl"))

        
    def load_model(self, load_path):
        logger.info(f"Loading model from {load_path}")
        self.model.from_pretrained(os.path.join(load_path, "checkpoint-bart"))
        self.roberta.load_state_dict(torch.load(os.path.join(load_path, "roberta.mdl"), map_location=f'cuda:{self.config.gpu_device}'))
        self.ffnn.load_state_dict(torch.load(os.path.join(load_path, "ffnn.mdl"), map_location=f'cuda:{self.config.gpu_device}'))
       

class AMRPrefixGen(AMRPrefixGenBase):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.config = config
        self.tokenizer = tokenizer
        config.model_name = config.pretrained_model_name
        logger.info(f'Using model {self.__class__.__name__}')
        logger.info(f'Loading pre-trained model {config.model_name}')

        if config.model_name.startswith('facebook/bart-'):
            # main model
            self.model_config =  AutoConfig.from_pretrained(config.model_name, cache_dir=config.cache_dir)
            self.model_config.output_attentions = True
            self.model_config.use_encoder_prefix = config.use_encoder_prefix
            self.model_config.use_cross_prefix = config.use_cross_prefix
            self.model_config.use_decoder_prefix = config.use_decoder_prefix
            self.model_config.prefix_length = config.prefix_length
            
            self.model = PrefixGenBartForConditionalGeneration.from_pretrained(config.model_name, cache_dir=config.cache_dir, config=self.model_config)
            

            self.roberta = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.2',cache_dir=config.cache_dir)
            self.prefix_tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2',cache_dir=config.cache_dir)
            self.label_tokens = ["<T>","<number>","<roles>"]
            self.prefix_tokenizer.add_tokens(self.label_tokens, special_tokens=True)
            self.roberta.resize_token_embeddings(len(self.prefix_tokenizer))
            self.num_layers = 12  
            self.head_num = 16  
            self.head_dim = 64  # obtain the demansion of every head
            #logger.info(f'num_layers: {self.num_layers},head_num: {self.head_num},head_dim: {self.head_dim}')
            self.ffnn = nn.Linear(self.roberta.config.hidden_size, 
                                    self.num_layers * self.config.prefix_length * self.head_num * self.head_dim)
            
            self.roberta = self.roberta.cuda()
            self.ffnn = self.ffnn.cuda()
           
        else:
            raise ValueError("Model does not support yet.")
        self.model.resize_token_embeddings(len(self.tokenizer))

        # if self.config.pretrained_model_path is not None:
        #     self.load_model(self.config.pretrained_model_path)


        if config.freeze_bart:
            for param in self.model.parameters():
                param.requires_grad=False