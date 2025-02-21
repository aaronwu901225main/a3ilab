from itertools import product
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import argparse
import clip
from collections import OrderedDict
from clip_modules.model_loader import load
from model.common import *
import numpy as np



class DFSP(nn.Module):
    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        clip_model, _ = load(config.clip_model, context_length=config.context_length)
        self.clip = clip_model
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.attr_dropout = nn.Dropout(config.attr_dropout)
        self.dropout = nn.Dropout(p=0.5)  ##新增Monte Carlo Dropout
        self.token_ids, self.soft_att_obj, ctx_vectors = self.construct_soft_prompt()
        self.offset = offset
        self.enable_pos_emb = True
        dtype = None
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.dtype)
        for p in self.parameters():
            p.requires_grad=False

        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.soft_prompt = nn.Parameter(ctx_vectors).cuda()
        self.fusion = FusionTextImageBlock(config.width_img, config.width_txt, len(self.attributes), len(self.classes), config.SA_K, context_length=self.config.context_length, fusion=self.config.fusion)
        self.weight = config.res_w
        self.log_idx = True

    def construct_soft_prompt(self):
        token_ids = clip.tokenize("a photo of x x",
                              context_length=self.config.context_length).cuda()

        tokenized = torch.cat(
            [
                clip.tokenize(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding = self.clip.token_embedding(tokenized.cuda())

        # with torch.no_grad():
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = "a photo of "
        n_ctx = len(ctx_init.split())
        prompt = clip.tokenize(ctx_init,
                            context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        return token_ids, soft_att_obj, ctx_vectors


    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.clip.token_embedding(
            class_token_ids.cuda()
        ).type(self.clip.dtype)
        soft_att_obj = self.attr_dropout(self.soft_att_obj)
        eos_idx = int(self.token_ids[0].argmax())
        token_tensor[:, eos_idx - 2, :] = soft_att_obj[
            attr_idx
        ].type(self.clip.dtype)
        token_tensor[:, eos_idx - 1, :] = soft_att_obj[
            obj_idx + self.offset
        ].type(self.clip.dtype)

        # adding the correct learnable context
        token_tensor[
            :, 1 : len(self.soft_prompt) + 1, :
        ] = self.soft_prompt.type(self.clip.dtype)
        return token_tensor



    def visual(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        img_feature = self.clip.visual.transformer(x)

        x = img_feature.permute(1, 0, 2)  # LND -> NLD

        x = self.clip.visual.ln_post(x[:, 0, :])
        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        return x, img_feature


    def ft_to_logit(self, img, txt):
        img_feature = img.permute(1, 0, 2)  # LND -> NLD

        img_feature = self.clip.visual.ln_post(img_feature[:, 0, :])
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj
        
        if self.config.fusion in ["BiFusion", "img2txt"]: 
            txt_feature = txt.permute(0, 2, 1, 3)
            txt_feature = self.text_encoder.ln_final(txt_feature)
            txt_tf = (
                txt_feature[
                    :, torch.arange(txt_feature.shape[1]), self.token_ids.argmax(dim=-1)
                ]  # POS of <EOS>
                @ self.text_encoder.text_projection
            )
        else:
            txt_feature = txt.permute(1, 0, 2)
            txt_feature = self.text_encoder.ln_final(txt_feature)
            txt_tf = (
                txt_feature[
                    torch.arange(txt_feature.shape[0]), self.token_ids.argmax(dim=-1)
                ]  # POS of <EOS>
                @ self.text_encoder.text_projection
            )

        return img_feature, txt_tf

    def decompose_logits(self, logits, idx):
        att_idx, obj_idx = idx[:, 0].cpu().numpy(), idx[:, 1].cpu().numpy()
        logits_att = torch.zeros(logits.shape[0], len(self.attributes)).cuda()
        logits_obj = torch.zeros(logits.shape[0], len(self.classes)).cuda()
        for i in range(len(self.attributes)):
            logits_att[:, i] = logits[:, np.where(att_idx==i)[0]].mean(-1)
        for i in range(len(self.classes)):
            logits_obj[:, i] = logits[:, np.where(obj_idx==i)[0]].mean(-1)     

        return logits_att, logits_obj

###---------------增加計算投影--------------
    def cal_proj_f_v_to_s_given_o(self,f_v,f_s_given_o):
        # 計算 f_s_given_o 的轉置乘以 f_s_given_o，再取其反矩陣
        temp = torch.matmul(f_s_given_o, f_s_given_o.t()).inverse()
        #最後與 𝑓(𝑠|𝑜)(𝑓_(𝑠|𝑜)^𝑇 𝑓_(𝑠|𝑜))〗^(−1)𝑓(𝑠|𝑜)^T
        temp1 = torch.matmul(f_s_given_o.t(),temp)
        temp2 =torch.matmul(temp1,f_s_given_o)
        # 計算 f_v 與上述結果的乘積，即可得到 f_v->s|o 的轉置
        f_v_to_s_given_o_t = torch.matmul(temp2, f_v.t().float())
        # 轉置回來即為 f_v->s|o
        f_v_to_s_given_o = f_v_to_s_given_o_t.t()
        return f_v_to_s_given_o




    def forward(self, batch_img, idx,know_obj_test = False,gt_obj_idx=None,all_fv_proj_idx=None,phase=False):
        b = batch_img.shape[0]
        l, _ = idx.shape
        batch_img, img_ft = self.visual(batch_img.type(self.clip.dtype))   ## bs * 768
        token_tensors = self.construct_token_tensors(idx)
        text_features, text_ft = self.text_encoder(
            self.token_ids,
            token_tensors,
            enable_pos_emb=self.enable_pos_emb,
        )  

        batch_img_soft_prompt = batch_img / batch_img.norm(dim=-1, keepdim=True)
        text_features_soft_prompt = text_features / text_features.norm(dim=-1, keepdim=True)
        img_ft, text_ft = self.fusion(img_ft.type(torch.float), text_ft.type(torch.float), idx, b)
        img_ft, text_ft = self.ft_to_logit(img_ft.type(self.clip.dtype), text_ft.type(self.clip.dtype))
        batch_img = self.weight * batch_img + (1 - self.weight) * img_ft
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
        if self.config.fusion in ["BiFusion", "img2txt"]: 
            text_features = self.weight * text_features.repeat(b, 1, 1) + (1 - self.weight) * text_ft
        else:
            text_features = self.weight * text_features + (1 - self.weight) * text_ft
        idx_text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )
#         if self.config.fusion in ["BiFusion", "img2txt"]: 
#             logits = (
#                 self.clip.logit_scale.exp()
#                 * normalized_img.unsqueeze(1)
#                 @ idx_text_features.permute(0,2,1)
#             ).squeeze()     ###     48 * 1262
#         else:
#             logits = (
#                 self.clip.logit_scale.exp()
#                 * normalized_img
#                 @ idx_text_features.t()
#             )   

#         logits_soft_prompt = (
#             self.clip.logit_scale.exp()
#             * batch_img_soft_prompt
#             @ text_features_soft_prompt.t()
#         )     

#         logits_att, logits_obj = self.decompose_logits(logits_soft_prompt, idx)
        if phase:
            num_samples = 10  # number of dropout samples to take
            logits_list = []
            for i in range(num_samples):
                if self.config.fusion in ["BiFusion", "img2txt"]: 
                    mc_normalized_img = self.dropout(normalized_img.unsqueeze(1))
                    logits = (
                        self.clip.logit_scale.exp()
                        * mc_normalized_img
                        @ idx_text_features.permute(0, 2, 1)
                    ).squeeze()  # 48 * 1262
                else:
                    mc_normalized_img = self.dropout(normalized_img)
                    logits = (
                        self.clip.logit_scale.exp() * mc_normalized_img @ idx_text_features.t()
                    )
                logits_list.append(logits)
            logits = torch.stack(logits_list).mean(dim=0)
        else:
            if self.config.fusion in ["BiFusion", "img2txt"]: 
                logits = (
                    self.clip.logit_scale.exp()
                    * normalized_img.unsqueeze(1)
                    @ idx_text_features.permute(0, 2, 1)
                ).squeeze()  # 48 * 1262
            else:
                logits = (
                    self.clip.logit_scale.exp() * normalized_img @ idx_text_features.t()
                )
                
        if know_obj_test:  ###是否要改logit_attrs
#             print(idx_text_features.size())
            ft_plus=self.fusion.decompose_ftfsfo(idx_text_features,idx)
            _,attr_nub=logits_att.size()
            fo = ft_plus[attr_nub:, :]
            fs = ft_plus[:attr_nub, :]   ###獲得fs
            
            f_s_given_o = torch.Tensor().cuda()
            proj_f_v_to_s_given_o_finally = torch.Tensor().cuda()
            for gt_obj,batch_idx in zip(gt_obj_idx,range(len(gt_obj_idx))):  ### gt_obj_idx=f=data[2]是當前batch的gt，gt_obj是當前的gt
                f_s_given_o = torch.Tensor().cuda()
                for proj_idx in all_fv_proj_idx[gt_obj]:       
                    ##將當前batch的第batch_idx個gt輸入進all_fv_proj_idx中，並找尋他的位子
                    index = torch.nonzero(torch.eq(idx, proj_idx).all(dim=1))         
                    f_s_given_o = torch.cat([f_s_given_o, idx_text_features[index[0]]], dim=0)   ###獲得fv_proj s|o
#                 print(f_s_given_o.size())
                proj_f_v_to_s_given_o = self.cal_proj_f_v_to_s_given_o(normalized_img[batch_idx],f_s_given_o)   
                ###第 batch_idx個圖片與f_s_given_o進行計算 ，獲得proj_f_v_to_s_given_o後 concat起來
                ### fv dim(b,768) f(𝑠|𝑜)  dim(?,768)>f(v→s|o)^T dim(b,768)
                proj_f_v_to_s_given_o_finally = torch.cat([proj_f_v_to_s_given_o_finally, proj_f_v_to_s_given_o.unsqueeze(0)] , dim=0)
            proj_f_v_to_s_given_o_finally_norm=proj_f_v_to_s_given_o_finally / proj_f_v_to_s_given_o_finally.norm(dim=-1, keepdim=True)

            logits_att_our = (
                self.clip.logit_scale.exp()
                * proj_f_v_to_s_given_o_finally_norm
                @ fs.t()
                        )   
            
        
        
        return (logits, logits_att, logits_obj, logits_soft_prompt,logits_att_our)