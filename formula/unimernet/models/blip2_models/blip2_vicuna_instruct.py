"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
from packaging import version

import torch
import torch.nn as nn

import transformers

from unimernet.common.registry import registry
from unimernet.models.blip2_models.blip2 import Blip2Base, disabled_train


@registry.register_model("blip2_vicuna_instruct")
class Blip2VicunaInstruct(Blip2Base):
    """
    BLIP2 Vicuna model.
    Supported model types:
        - vicuna7b
        - vicuna13b
    Usage:
        >>> from unimernet.models import load_model
        >>> model = load_model("blip2_vicuna_instruct", "vicuna7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2_instruct_vicuna7b.yaml",
        "vicuna13b": "configs/models/blip2_instruct_vicuna13b.yaml",
        "minigpt4_vicuna7b": "configs/models/mini_gpt4_vicuna7b.yaml",
        "minigpt4_vicuna13b": "configs/models/mini_gpt4_vicuna13b.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            freeze_vit_ln=False,
            num_query_token=32,
            llm_model="",
            prompt="",
            max_txt_len=128,
            max_output_txt_len=256,
            apply_lemmatizer=False,
            qformer_text_input=True,
            truncate_q_former_output=True
    ):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.28"), "BLIP-2 Vicuna requires transformers>=4.28"
        from transformers import LlamaTokenizer
        from unimernet.models.blip2_models.modeling_llama import LlamaForCausalLM

        self.tokenizer = self.init_tokenizer(truncation_side="left")

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        if freeze_vit_ln:
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vit layner norm")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
        self.llm_tokenizer_for_generate = LlamaTokenizer.from_pretrained(llm_model, use_fast=False,
                                                                         truncation_side="left")
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16
        )
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_tokenizer_for_generate.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer_for_generate.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer_for_generate.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer_for_generate.add_special_tokens({'unk_token': '</s>'})
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # self.eos_token_id = self.llm_tokenizer(
        #     self.llm_tokenizer.eos_token, add_special_tokens=False
        # ).input_ids[0]

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input
        self.truncate_q_former_output = truncate_q_former_output

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def forward(self, samples):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')

        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        bs = image.size(0)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                samples["text_input"],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        if self.truncate_q_former_output:
            inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        else:
            inputs_llm = self.llm_proj(query_output.last_hidden_state)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(image.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                use_cache=False,
            )

        loss = outputs.loss

        return {"loss": loss}

    def get_vision_feats(self, image, prompt):
        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        query_tokens = self.query_tokens.expand(bs, -1, -1)

        text_Qformer = self.tokenizer(
            prompt,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_output = self.Qformer.bert(
            text_Qformer.input_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        if self.truncate_q_former_output:
            inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        else:
            inputs_llm = self.llm_proj(query_output.last_hidden_state)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llm, atts_llm

    def shift_padding_to_left(self, inputs_embeds, attention_mask):
        llm_tokens = {"input_embeds": [], "attention_mask": []}
        for i in range(inputs_embeds.size(0)):
            this_input_ones = attention_mask[i].sum()
            llm_tokens['input_embeds'].append(
                torch.cat([
                    inputs_embeds[i][this_input_ones:],
                    inputs_embeds[i][:this_input_ones],
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    attention_mask[i][this_input_ones:],
                    attention_mask[i][:this_input_ones],
                ])
            )
        llm_tokens['input_embeds'] = torch.stack(llm_tokens['input_embeds'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens['input_embeds'], llm_tokens['attention_mask']

    @torch.no_grad()
    def generate(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1,
            num_captions=1,
            temperature=1,
    ):

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        image = samples["image"]

        inputs_llm, atts_llm = self.get_vision_feats(image, prompt)

        self.llm_tokenizer_for_generate.padding_side = "right"

        self.llm_tokenizer_for_generate.pad_token = self.llm_tokenizer_for_generate.eos_token  # debug
        ori_pad_token_id = self.llm_model.config.pad_token_id
        self.llm_model.config.pad_token_id = self.llm_model.config.eos_token_id  # debug

        if "prefix" in samples:
            prompt = [f"{prompt_} {prefix_}".strip() for prompt_, prefix_ in zip(prompt, samples["prefix"])]

        llm_tokens = self.llm_tokenizer_for_generate(
            prompt,
            padding="longest",
            return_tensors="pt",
        ).to(image.device)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        inputs_embeds = inputs_embeds.to(next(self.llm_model.parameters()).dtype)
        attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)
        inputs_embeds, attention_mask = self.shift_padding_to_left(inputs_embeds, attention_mask)

        with self.maybe_autocast():
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                use_cache=True
            )

        outputs[outputs == 0] = 2  # convert output id 0 to 2 (eos_token_id)
        outputs[outputs == -1] = 1  # debug
        output_text = self.llm_tokenizer_for_generate.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        self.llm_model.config.pad_token_id = ori_pad_token_id

        return output_text

    @torch.no_grad()
    def generate_multi(
            self,
            samples,
            use_nucleus_sampling=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1,
            temperature=1,
    ):

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        image = samples["image"]

        inputs_llm, atts_llm = self.get_vision_feats(image, prompt)

        self.llm_tokenizer_for_generate.padding_side = "right"

        self.llm_tokenizer_for_generate.pad_token = self.llm_tokenizer_for_generate.eos_token  # debug
        ori_pad_token_id = self.llm_model.config.pad_token_id
        self.llm_model.config.pad_token_id = self.llm_model.config.eos_token_id  # debug

        if "prefix" in samples:
            prompt = [f"{prompt_} {prefix_}".strip() for prompt_, prefix_ in zip(prompt, samples["prefix"])]

        llm_tokens = self.llm_tokenizer_for_generate(
            prompt,
            padding="longest",
            return_tensors="pt",
        ).to(image.device)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        inputs_embeds = inputs_embeds.to(next(self.llm_model.parameters()).dtype)
        attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)
        inputs_embeds, attention_mask = self.shift_padding_to_left(inputs_embeds, attention_mask)

        with self.maybe_autocast():
            raw_output = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=True
            )
        outputs = raw_output.sequences
        outputs[outputs == 0] = 2  # convert output id 0 to 2 (eos_token_id)
        outputs[outputs == -1] = 1  # debug
        output_text = self.llm_tokenizer_for_generate.batch_decode(outputs, skip_special_tokens=True)

        output_text = [text.strip() for text in output_text]
        scores = torch.exp(raw_output.sequences_scores).cpu().numpy() ** 3 * 100  # TODO

        all_texts = []
        all_scores = []
        for i in range(0, len(output_text), num_beams):
            this_text = output_text[i:i + num_beams]
            all_texts.append(this_text)
            this_score = scores[i: i + num_beams]
            all_scores.append(this_score)

        self.llm_model.config.pad_token_id = ori_pad_token_id

        return all_texts, all_scores

    def predict_by_rank(
            self,
            samples,
            **kwargs
    ):
        image = samples["image"]
        prompt = samples["prompt"]
        candidates = samples["candidates"][0]
        if isinstance(prompt, str):
            prompt = [prompt]
        assert image.size(0) == len(prompt) == 1, "When doing predict by rank, the batch size must be 1."

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        batch_size = len(candidates)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        if self.truncate_q_former_output:
            inputs_llm = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        else:
            inputs_llm = self.llm_proj(query_output.last_hidden_state)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        inputs_llm = inputs_llm.repeat(batch_size, 1, 1)
        atts_llm = atts_llm.repeat(batch_size, 1)
        text_input_ids = text_input_tokens.input_ids.repeat(batch_size, 1)
        text_input_mask = text_input_tokens.attention_mask.repeat(batch_size, 1)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in candidates],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(image.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_ids,
            text_input_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                reduction="none",
                use_cache=False
            )

        loss = outputs.loss.view(batch_size)
        top1 = int(torch.argsort(loss, dim=-1)[0])

        return [candidates[top1]]

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llm_model = cfg.get("llm_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_vit_ln = cfg.get("freeze_vit_ln", False)
        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_text_input = cfg.get("qformer_text_input", True)
        truncate_q_former_output = cfg.get("truncate_q_former_output", True)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_vit_ln=freeze_vit_ln,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
            truncate_q_former_output=truncate_q_former_output
        )

        model.load_checkpoint_from_config(cfg)

        return model
