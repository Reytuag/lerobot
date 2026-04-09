from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLMForConditionalGeneration,
)

from torch import Tensor, nn
import copy
import torch

def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class Encoder(nn.Module):
    def __init__(self,model_id,expert_width_multiplier=1,num_expert_layers=4):
        super().__init__()
        config = AutoConfig.from_pretrained(model_id)
        # print(config)
        lm_expert_config = copy.deepcopy(config.text_config)
        hidden_size = lm_expert_config.hidden_size
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)  # hidden_size // 2
        lm_expert_config.intermediate_size = get_intermediate_size(int(hidden_size * expert_width_multiplier))
        # lm_expert_config.num_hidden_layers = self.num_vlm_layers
        if num_expert_layers > 0:
            lm_expert_config.num_hidden_layers = num_expert_layers
        self.model = AutoModel.from_config(lm_expert_config)
        self.rl_token_embed=nn.Parameter(torch.zeros(1,hidden_size,dtype=torch.bfloat16))

    def forward(self,embeddings):
        
        embeddings=torch.cat([embeddings,self.rl_token_embed.expand(embeddings.shape[0],-1,-1)],dim=1)
        embeds=self.model(inputs_embeds=embeddings,
            attention_mask=None,
            position_ids=None,
            use_cache=False)
        return(embeds.last_hidden_state[:,-1,:])


class Decoder(nn.Module):
    def __init__(self,model_id,expert_width_multiplier=1,num_expert_layers=4):
        super().__init__()
        config = AutoConfig.from_pretrained(model_id)
        # print(config)
        lm_expert_config = copy.deepcopy(config.text_config)
        hidden_size = lm_expert_config.hidden_size
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)  # hidden_size // 2
        lm_expert_config.intermediate_size = get_intermediate_size(int(hidden_size * expert_width_multiplier))
        # lm_expert_config.num_hidden_layers = self.num_vlm_layers
        if num_expert_layers > 0:
            lm_expert_config.num_hidden_layers = num_expert_layers
        self.model = AutoModel.from_config(lm_expert_config)

    def forward(self,embeddings,rl_embed):
        
        embeddings=torch.roll(embeddings,shifts=1,dims=1)
        embeddings[:,0,:]=rl_embed
        embeds=self.model(inputs_embeds=embeddings,
            attention_mask=None,
            position_ids=None,
            use_cache=False)
        return(embeds.last_hidden_state)



class Autoencoder(nn.Module):
    def __init__(self,model_id,expert_width_multiplier=1,num_expert_layers=4):
        super().__init__()
        self.encoder=Encoder(model_id,expert_width_multiplier,num_expert_layers)
        self.decoder=Decoder(model_id,expert_width_multiplier,num_expert_layers)

    def forward(self,embeddings):
        rl_embed=self.encoder(embeddings)
        embeds=self.decoder(embeddings,rl_embed)

        loss=nn.MSELoss()(embeddings,embeds)
        return loss
    

    def encode(self,embeddings):
        return self.encoder(embeddings)

# encoder=Encoder("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

# embed=torch.ones(1,64,960,dtype=torch.bfloat16)

# a=encoder.forward(embed)

# print(a.shape)


# decoder=Decoder("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

# b=decoder.forward(embed,a)

# print(b.shape)



# autoencoder=Autoencoder("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
# c=autoencoder.forward(embed)
# print(c.shape)
# print(c)
# d=autoencoder.encode(embed)
# print(d.shape)