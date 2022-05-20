from random import choice, choices, shuffle, randint
import random
import torch 
from param import CONFIG

PAD_ID = -1

class EntityDataset:
    def __init__(self, sents, tags, tokenizer, tag_mapping, is_train=False, augmentation_ratio=0.35):
        self.sents = sents
        self.tags = tags
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.augmentation_ratio = augmentation_ratio
        self.short_sent_ids = [idx for (idx, sent) in enumerate(self.sents) if (len(sent) < 30)]
        self.long_sent_ids = [idx for (idx, sent) in enumerate(self.sents) if (len(sent) > 70)]
        self.tag_mapping = tag_mapping
        
    def __len__(self):
        return len(self.sents)
    
    def __getitem__(self, ind):
        """
        Output: dict of  seq_len, token_ids, mask, token_type, tar_tag
        """
        global CONFIG
        config = CONFIG
        MAX_LEN = config.MAX_LEN
        sent = self.sents[ind]
        tag = self.tags[ind]
        
        # if random.random() < self.augmentation_ratio and self.is_train:
        #     sent, tag = self.augmentation()
        
        token_ids = []
        tar_tag = []
        
        for i, word in enumerate(sent):
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            token_len = len(tokens)
            token_ids.extend(tokens)
            tar_tag.extend([tag[i]]*token_len)
            
        #### bos_token: 0, eos_token:2
        token_ids = [0] + token_ids[:MAX_LEN-2] + [2]
        seq_len = len(token_ids)
        tar_tag = [self.tag_mapping['O']] + tar_tag[:MAX_LEN-2] + [self.tag_mapping['O']]
        
        padding_len = MAX_LEN - seq_len
        mask = [1]*seq_len + [0]*padding_len
        token_ids = token_ids + [1]*padding_len
        tar_tag = tar_tag + [PAD_ID]*padding_len
        token_type_ids = [0]*len(token_ids)
        
        return {
            'src_sent': ' '.join(sent),
            'ids': torch.tensor(token_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'tar_tag': torch.tensor(tar_tag, dtype=torch.long),
            'seq_len': seq_len
        }
    
    def augmentation(self):
        ## Concate 2 short sentences to new sentence
        if random.random() < 0.5:
            sent1_id, sent2_id = choices(self.short_sent_ids, k=2)
            sent1, sent2 = self.sents[sent1_id], self.sents[sent2_id]
            tag1, tag2 = self.tags[sent1_id], self.tags[sent2_id]
            sent = sent1 + sent2
            tag = tag1 + tag2
            return sent, tag
        
        else: # crop a long sentence to new sentence
            long_id = choice(self.long_sent_ids)
            sent = self.sents[long_id]
            tag = self.tags[long_id]

            start_id = randint(0, len(tag)//3)
            end_id = randint(len(tag)//2, len(tag))
            sent = sent[start_id:end_id]
            tag = tag[start_id:end_id]
            return sent, tag
            
    # def sentence_to_input(sent, device=torch.device('cuda')):
    #     """
    #     Convert text input to list of token_id with additinal info
    #     Input: Sentence in string format
    #     Output: token_ids, mask, token_type_ids, offset_map, seq_len
    #     """
    #     global config
    #     MAX_LEN = config.MAX_LEN
    #     input = self.tokenizer.encode_plus(sent, add_special_tokens=True, return_offsets_mapping=True, max_length=MAX_LEN, padding='max_length')
    #     token_ids = input['input_ids']
    #     mask = input['attention_mask']
    #     offset_map = input['offset_mapping']
            
    #     token_type_ids = [0]*len(token_ids)
    #     token_ids = torch.tensor(token_ids, dtype=torch.long)
    #     mask = torch.tensor(mask, dtype=torch.long)
    #     token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    #     real_seq_len = int(mask.sum())

    #     token_ids = token_ids.unsqueeze(dim=0).to(device)
    #     mask = mask.unsqueeze(dim=0).to(device)
    #     token_type_ids = token_type_ids.unsqueeze(dim=0).to(device)
    #     return token_ids, mask, token_type_ids, offset_map, real_seq_len
    
    # def decode_output(output, offset, seq_len):
    #     output = output.squeeze()[:seq_len]
    #     offset = offset[:seq_len]
    #     filted_output = [output[1]]
        
    #     for i in range(1, seq_len-1):
    #         pre_word_ind = offset[i-1]
    #         cur_word_ind = offset[i]
    #         if cur_word_ind[0] != pre_word_ind[1]:
    #             filted_output.append(output[i])
    #     return filted_output
