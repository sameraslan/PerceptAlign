'''
Adapted from `https://github.com/v-iashin/SpecVQGAN`.
Modified by Samer Aslan, 2023.
'''

import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf.listconfig import ListConfig

sys.path.insert(0, '.')  # nopep8
from specvqgan.modules.transformer.mingpt import (GPTClass, GPTFeats, GPTFeatsClass)
from train import instantiate_from_config


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

# Feature extractor for audio and video featues to compute self-synchronization loss
class FeatureExtractor(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.gelu1 = nn.GELU()

        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(output_channels)
        self.gelu2 = nn.GELU()

        self.fc = nn.Linear(output_channels, output_channels)

    def forward(self, x):
        # First conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu1(x)

        # Second conv
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu2(x)

        x = x.permute(0, 2, 1)  # Reshape for the linear layer
        x = self.fc(x)
        x = x.permute(0, 2, 1)  # Reshape back

        # Apply L2 normalization for normalized local embeddings
        x = F.normalize(x, p=2, dim=1)

        return x


class Net2NetTransformer(pl.LightningModule):
    def __init__(self, hyperparameters, transformer_config, first_stage_config,
                 cond_stage_config,
                 first_stage_permuter_config=None, cond_stage_permuter_config=None,
                 ckpt_path=None, ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 downsample_cond_size=-1,
                 pkeep=1.0):

        super().__init__()

        # Initialize feature extractors
        self.audio_feature_extractor = FeatureExtractor(input_channels=80, output_channels=128)
        self.video_feature_extractor = FeatureExtractor(input_channels=2048, output_channels=128)

        if hyperparameters is not None:
            # Extract hyperparameters
            self.K = hyperparameters.get('K', 10)
            self.tau = hyperparameters.get('tau', 1)
            self.offset = hyperparameters.get('offset', 1)
            self.segments = hyperparameters.get('segments', 16)
            self.pal_weight = hyperparameters.get('pal_weight', 1.0)
            self.ce_weight = hyperparameters.get('ce_weight', 1.0)
            
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(cond_stage_config)
        if first_stage_permuter_config is None:
            first_stage_permuter_config = {"target": "specvqgan.modules.transformer.permuter.Identity"}
        if cond_stage_permuter_config is None:
            cond_stage_permuter_config = {"target": "specvqgan.modules.transformer.permuter.Identity"}
        self.first_stage_permuter = instantiate_from_config(config=first_stage_permuter_config)
        self.cond_stage_permuter = instantiate_from_config(config=cond_stage_permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def init_cond_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.cond_stage_model = model

    def forward(self, x, c):
        # one step to produce the logits
        quant_z, z_indices = self.encode_to_z(x)
        _, c_indices = self.encode_to_c(c)

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep * torch.ones(z_indices.shape, device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices

        # in the case we do not want to encode condition anyhow (e.g. inputs are features)
        if isinstance(self.transformer, (GPTFeats, GPTClass, GPTFeatsClass)):
            # make the prediction
            logits, _, _ = self.transformer(z_indices[:, :-1], c)
            # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
            if isinstance(self.transformer, GPTFeatsClass):
                cond_size = c['feature'].size(-1) + c['target'].size(-1)
            else:
                cond_size = c.size(-1)
            logits = logits[:, cond_size-1:]
        else:
            cz_indices = torch.cat((c_indices, a_indices), dim=1)
            # make the prediction
            logits, _, _ = self.transformer(cz_indices[:, :-1])
            # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
            logits = logits[:, c_indices.shape[1]-1:]

        return logits, target, quant_z

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):
        x = x if isinstance(self.transformer, (GPTFeats, GPTClass, GPTFeatsClass)) else torch.cat((c, x), dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            raise NotImplementedError('Implement for GPTFeatsCLass')
            raise NotImplementedError('Implement for GPTFeats')
            raise NotImplementedError('Implement for GPTClass')
            raise NotImplementedError('also the model outputs attention')
            # one pass suffices since input is pure noise anyway
            assert len(x.shape)==2
            # noise_shape = (x.shape[0], steps-1)
            # noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:,x.shape[1]-c.shape[1]:-1]
            x = torch.cat((x,noise),dim=1)
            logits, _ = self.transformer(x)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0]*shape[1],shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0],shape[1],shape[2])
                ix = ix.reshape(shape[0],shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            x = ix[:, c.shape[1]-1:]
        else:
            for k in range(steps):
                callback(k)
                if isinstance(self.transformer, (GPTFeats, GPTClass, GPTFeatsClass)):
                    # if assert is removed, you need to make sure that the combined len is not longer block_s
                    if isinstance(self.transformer, GPTFeatsClass):
                        cond_size = c['feature'].size(-1) + c['target'].size(-1)
                    else:
                        cond_size = c.size(-1)
                    assert x.size(1) + cond_size <= block_size

                    x_cond = x
                    c_cond = c
                    logits, _, att = self.transformer(x_cond, c_cond)
                else:
                    assert x.size(1) <= block_size  # make sure model can see conditioning
                    x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                    logits, _, att = self.transformer(x_cond)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
            # cut off conditioning
            x = x if isinstance(self.transformer, (GPTFeats, GPTClass, GPTFeatsClass)) else x[:, c.shape[1]:]
        return x, att.detach().cpu()

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.first_stage_permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, info = self.cond_stage_model.encode(c)
        if isinstance(self.transformer, (GPTFeats, GPTClass, GPTFeatsClass)):
            # these are not indices but raw features or a class
            indices = info[2]
        else:
            indices = info[2].view(quant_c.shape[0], -1)
            indices = self.cond_stage_permuter(indices)
        return quant_c, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape, stage='first'):
        if stage == 'first':
            index = self.first_stage_permuter(index, reverse=True)
        elif stage == 'cond':
            print('in cond stage in decode_to_img which is unexpected ')
            index = self.cond_stage_permuter(index, reverse=True)
        else:
            raise NotImplementedError

        bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        log = dict()

        N = 4
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c = self.get_xc(batch, N)
        x = x.to(device=self.device)
        # c = c.to(device=self.device)
        if isinstance(c, dict):
            c = {k: v.to(self.device) for k, v in c.items()}
        else:
            c = c.to(self.device)

        quant_z, z_indices = self.encode_to_z(x)
        quant_c, c_indices = self.encode_to_c(c)  # output can be features or a single class or a featcls dict

        # create a "half"" sample
        z_start_indices = z_indices[:, :z_indices.shape[1]//2]
        index_sample, att_half = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1]-z_start_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample = self.decode_to_img(index_sample, quant_z.shape)

        # sample
        z_start_indices = z_indices[:, :0]
        index_sample, att_nopix = self.sample(z_start_indices, c_indices,
                                              steps=z_indices.shape[1],
                                              temperature=temperature if temperature is not None else 1.0,
                                              sample=True,
                                              top_k=top_k if top_k is not None else 100,
                                              callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)

        # det sample
        z_start_indices = z_indices[:, :0]
        index_sample, att_det = self.sample(z_start_indices, c_indices,
                                            steps=z_indices.shape[1],
                                            sample=False,
                                            callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)

        log["inputs"] = x
        log["reconstructions"] = x_rec

        if isinstance(self.cond_stage_key, str):
            cond_is_not_image = self.cond_stage_key != "image"
            cond_has_segmentation = self.cond_stage_key == "segmentation"
        elif isinstance(self.cond_stage_key, ListConfig):
            cond_is_not_image = 'image' not in self.cond_stage_key
            cond_has_segmentation = 'segmentation' in self.cond_stage_key
        else:
            raise NotImplementedError

        if cond_is_not_image:
            cond_rec = self.cond_stage_model.decode(quant_c)
            if cond_has_segmentation:
                # get image from segmentation mask
                num_classes = cond_rec.shape[1]

                c = torch.argmax(c, dim=1, keepdim=True)
                c = F.one_hot(c, num_classes=num_classes)
                c = c.squeeze(1).permute(0, 3, 1, 2).float()
                c = self.cond_stage_model.to_rgb(c)

                cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)
                cond_rec = F.one_hot(cond_rec, num_classes=num_classes)
                cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()
                cond_rec = self.cond_stage_model.to_rgb(cond_rec)
            log["conditioning_rec"] = cond_rec
            log["conditioning"] = c

        log["samples_half"] = x_sample
        log["samples_nopix"] = x_sample_nopix
        log["samples_det"] = x_sample_det
        log["att_half"] = att_half
        log["att_nopix"] = att_nopix
        log["att_det"] = att_det
        return log

    def get_input(self, key, batch):
        if isinstance(key, str):
            # if batch[key] is 1D; else the batch[key] is 2D
            if key in ['feature', 'target']:
                x = self.cond_stage_model.get_input(batch, key)
            else:
                x = batch[key]
                if len(x.shape) == 3:
                    x = x[..., None]
                x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            if x.dtype == torch.double:
                x = x.float()
        elif isinstance(key, ListConfig):
            x = self.cond_stage_model.get_input(batch, key)
            for k, v in x.items():
                if v.dtype == torch.double:
                    x[k] = v.float()
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            if isinstance(self.cond_stage_key, ListConfig):
                c = {k: v[:N] for k, v in c.items()}
            else:
                c = c[:N]
        return x, c


    def compute_cross_correlation(self, audio_emb, video_emb, K, tau, segments):
        batch_size, _, Tm = audio_emb.size()
        segment_size = Tm // segments

        # Reshape embeddings to incorporate segments
        audio_emb_reshaped = audio_emb.unfold(2, segment_size, segment_size)
        video_emb_reshaped = video_emb.unfold(2, segment_size, segment_size)

        # Initialize cross-correlation tensor
        cc_total = torch.zeros(batch_size, segments, 2 * K + 1, device=audio_emb.device)

        for k in range(-K, K + 1):
            # Shift and pad video embeddings for each segment
            shifted_video_emb = F.pad(video_emb_reshaped, (K, K), "constant", 0)
            shifted_video_emb = shifted_video_emb[:, :, :, K - k:K - k + segment_size]

            # Compute cross-correlation for each segment
            cc = (shifted_video_emb * audio_emb_reshaped).sum(dim=3).sum(dim=1)
            cc_total[:, :, k] = cc

        # Apply softmax to the cross-correlation values for each segment
        cc_total = F.softmax(cc_total / tau, dim=2)

        return cc_total


    def compute_pal_loss(self, audio_emb, video_emb, K, tau, segments, offsets):
        cc_total = self.compute_cross_correlation(audio_emb, video_emb, K, tau, segments)
        
        # Take offsets within weighted window
        if offsets > 0:
            # Create symmetric weights centered around zero
            weights = torch.exp(-torch.arange(-offsets, offsets + 1, dtype=torch.float32) ** 2 / (2 * (offsets / 2) ** 2))
            weights = weights / weights.sum()  # Normalize weights
            weights = weights.to(cc_total.device)  # Ensure weights are on the same device

            # Apply weighted sum for each segment
            start_index = K - offsets
            end_index = K + offsets + 1
            cc_segmented = cc_total[:, :, start_index:end_index]
            weighted_sums = torch.einsum('bsk,k->bs', cc_segmented, weights)
        else:
            # Directly use the probability at zero offset
            weighted_sums = cc_total[:, :, K]

        # Averaging (weighted sums or zero offset probabilities) across segments
        offset_distribution = weighted_sums.mean(dim=1)

        # Compute pal loss
        pal_loss = -torch.log(offset_distribution + 1e-9)  # Added small value to prevent log(0)
        pal_loss = pal_loss.mean()

        return pal_loss

    
    def decode_logits_to_mel_spectrogram(self, logits, quant_z_shape):
        # Get the predicted code sequence by taking the max value at each point in the sequence
        _, predicted_indices = torch.max(logits, dim=-1)

        # Decode the predicted code sequence back to mel spectrogram
        mel_spectrogram = self.decode_to_img(predicted_indices, zshape=quant_z_shape, stage='first')

        # Ensure the mel_spectrogram is of the expected shape [batch_size, 1, 80, 848]
        if mel_spectrogram.shape != (logits.shape[0], 1, 80, 848):
            raise ValueError(f"Decoded mel spectrogram has unexpected shape: {mel_spectrogram.shape}")

        return mel_spectrogram


    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)

        logits, target, quant_z = self(x, c)

        mel_pred = self.decode_logits_to_mel_spectrogram(logits, quant_z.shape)  # Get predicted spectrogram


        mel_pred = mel_pred.squeeze(1)  # Remove the second dimension of length 1

        # Linearly upsample video features (c) to match audio codebook features (x)'s temporal dimension
        c_resampled = F.interpolate(c, size=(x.size(3)), mode='linear', align_corners=False)

        # Get the single layer embeddings
        audio_emb = self.audio_feature_extractor(mel_pred)  # Get the audio embeddings from the predicted mel spectrogram
        video_emb = self.video_feature_extractor(c_resampled)
        
        # Compute PAL loss
        pal_loss = self.compute_pal_loss(audio_emb, video_emb, self.K, self.tau, self.segments, self.offset)

        # Compute CE loss
        ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        
        # Combine the losses
        total_loss = self.ce_weight * ce_loss + self.pal_weight * pal_loss

        return total_loss


    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )

        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.Conv1d, torch.nn.LSTM, torch.nn.GRU)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif ('weight' in pn or 'bias' in pn) and isinstance(m, (torch.nn.LSTM, torch.nn.GRU)):
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer


if __name__ == '__main__':
    from omegaconf import OmegaConf

    cfg_image = OmegaConf.load('./configs/vggsound_transformer.yaml')
    cfg_image.model.params.first_stage_config.params.ckpt_path = './logs/2021-05-19T22-16-54_vggsound_specs_vqgan/checkpoints/epoch_39.ckpt'

    transformer_cfg = cfg_image.model.params.transformer_config
    first_stage_cfg = cfg_image.model.params.first_stage_config
    cond_stage_cfg = cfg_image.model.params.cond_stage_config
    permuter_cfg = cfg_image.model.params.permuter_config
    hyper_cfg = cfg_image.model.params.hyperparameters
    
    transformer = Net2NetTransformer(
        transformer_cfg, first_stage_cfg, cond_stage_cfg, hyper_cfg, permuter_cfg
    )

    c = torch.rand(2, 2048, 212)
    x = torch.rand(2, 1, 80, 848)

    logits, target = transformer(x, c)
    print(logits.shape, target.shape)
