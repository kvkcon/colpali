from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM, StoppingCriteria, TextStreamer, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
import yaml
from datasets import Dataset, load_dataset, Features, Value, Sequence, Image, load_from_disk
from transformers.cache_utils import Cache
import sys
sys.path.insert(0, '/home/jinaai/binwenbai/rerank_vlm/')
from ocr.got_vision_b import build_GOT_vit_b

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import List, cast, Optional, Tuple, Union
import requests
from io import BytesIO

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import dataclasses
from tqdm import tqdm


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

from enum import auto, Enum
class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()

class GOTImageEvalProcessor:
    def __init__(self, image_size=384, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
    def __call__(self, item):
        return self.transform(item)

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "<|im_end|>"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep + '\n'
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        if self.sep_style == SeparatorStyle.MPT:
            if self.system:
                ret = self.system + self.sep 
            else:
                ret = ''
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")


    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2)

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
        self.keyword_ids = [keyword_id[0] for keyword_id in self.keyword_ids if type(keyword_id) is list and len(keyword_id) == 1]
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            for keyword_id in self.keyword_ids:
                if output_ids[0, -1] == keyword_id:
                    return True
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
        self.keyword_ids = [keyword_id[0] for keyword_id in self.keyword_ids if type(keyword_id) is list and len(keyword_id) == 1]
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            for keyword_id in self.keyword_ids:
                if output_ids[0, -1] == keyword_id:
                    return True
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False
    

class GOTImageEvalProcessor:
    def __init__(self, image_size=384, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
    def __call__(self, item):
        return self.transform(item)


class GOTConfig(Qwen2Config):
    model_type = "GOT"


class GOTQwenModel(Qwen2Model):
    config_class = GOTConfig

    def __init__(self, config: Qwen2Config):
        super(GOTQwenModel, self).__init__(config)

        self.vision_tower_high = build_GOT_vit_b()

        self.mm_projector_vary =  nn.Linear(1024, 1024)


    def initialize_vision_modules(
        self, 
        vision_tower,
        pretrained_stage1_model=None,
        freeze_vision_tower=False,
        use_im_start_end=False,
        vision_select_layer=-1,
        dtype=torch.float16,
        device="cuda"
    ):


        image_processor_high = GOTImageEvalProcessor(image_size=1024)
      
        self.vision_tower_high = self.vision_tower_high.to(dtype=dtype, device=device)

        self.mm_projector_vary = self.mm_projector_vary.to(dtype=dtype, device=device)


        image_token_len = 256

        self.config.vision_tower = vision_tower
        self.config.image_token_len = image_token_len

        self.config.use_im_start_end = True

        self.config.vision_select_layer = vision_select_layer
        self.config.freeze_vision_tower = freeze_vision_tower
        
        return dict(
            image_processor_high=image_processor_high,
            image_token_len=image_token_len,
        )
         
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        if orig_embeds_params is not None:
            with torch.no_grad():
                self.get_input_embeddings().weight[:-self.num_new_tokens] = orig_embeds_params[:-self.num_new_tokens].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)


        vision_tower_high = getattr(self, 'vision_tower_high', None)


        if vision_tower_high is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:
            use_im_start_end = getattr(self.config, "use_im_start_end", -1)

            vision_select_layer = getattr(self.config, "vision_select_layer", -1)
            im_patch_token = getattr(self.config, "im_patch_token", -1)
            im_start_token = getattr(self.config, "im_start_token", -1)
            im_end_token = getattr(self.config, "im_end_token", -1)
            freeze_vision_tower = getattr(self.config, "freeze_vision_tower", False)

            im_patch_token = 151859

            im_start_token = 151857

            im_end_token = 151858
            
            image_features = []
            
            for image in images:
                P, C, H, W = image.shape
                if P == 1:
                    with torch.set_grad_enabled(False):
                        cnn_feature = vision_tower_high(image)
                        cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1) # 256*1024
                    image_feature = self.mm_projector_vary(cnn_feature)
                    image_features.append(image_feature)

                else:
                    image_patches = torch.unbind(image)
                    image_patches_features = []
                    for image_patch in image_patches:
                        image_p = torch.stack([image_patch])
                        
                        with torch.set_grad_enabled(False):
                            cnn_feature_p = vision_tower_high(image_p)
                            cnn_feature_p = cnn_feature_p.flatten(2).permute(0, 2, 1)
                        image_feature_p = self.mm_projector_vary(cnn_feature_p)
                        image_patches_features.append(image_feature_p)
                    image_feature = torch.cat(image_patches_features, dim=1)
                    image_features.append(image_feature)


            dummy_image_features_2 = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features = dummy_image_features_2
            use_im_start_end = True
            new_input_embeds = []
            for cur_input_ids, cur_input_embeds, cur_image_features in zip(input_ids, inputs_embeds, image_features):
                if (cur_input_ids == im_patch_token).sum() == 0:
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue

                if use_im_start_end:
                    if (cur_input_ids == im_start_token).sum() != (cur_input_ids == im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    
                    image_start_tokens = torch.where(cur_input_ids == im_start_token)[0]
                    for image_start_token_pos, per_cur_image_features in zip(image_start_tokens, cur_image_features):
                        per_cur_image_features = per_cur_image_features.to(device=cur_input_embeds.device)
                        num_patches = per_cur_image_features.shape[0]

                        if cur_input_ids[image_start_token_pos + num_patches + 1] != im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        
                        cur_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:image_start_token_pos+1], 
                                per_cur_image_features, 
                                cur_input_embeds[image_start_token_pos + num_patches + 1:]
                            ), 
                            dim=0
                        )


                    new_input_embeds.append(cur_input_embeds)
                else:
                    raise NotImplementedError

            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(GOTQwenModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, position_ids = position_ids,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

class OcrHfDatasetsModel(Qwen2ForCausalLM):
    config_class = GOTConfig
    # supports_gradient_checkpointing = True

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = GOTQwenModel(config)

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs  = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            return_dict=return_dict
            
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(
        self, 
        tokenizer, 
        freeze_lm_model=False, 
        pretrained_stage1_model=None,
        device="cuda"
    ):
        config = self.get_model().config


        self.resize_token_embeddings(len(tokenizer))

        config.im_patch_token = 151859

        config.use_im_start_end = True

        if config.use_im_start_end:
            self.resize_token_embeddings(len(tokenizer))
            config.im_start_token, config.im_end_token = 151857, 151858

    def load_image(self, image_file):
        if isinstance(image_file, str):
            if image_file.startswith('http') or image_file.startswith('https'):
                response = requests.get(image_file)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_file).convert('RGB')
            return image
        else:
            if image_file.mode != 'RGB':
                image_file = image_file.convert('RGB')
            return image_file

    def disable_torch_init(self):
        """
        Disable the redundant torch default initialization to accelerate model creation.
        """
        import torch
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

    def chat(self, tokenizer, image_file, ocr_type, ocr_box='', ocr_color='', render=False, save_render_file=None, print_prompt=False, gradio_input=False, stream_flag = False):

        self.disable_torch_init()


        image_processor_high =  GOTImageEvalProcessor(image_size=1024)

        use_im_start_end = True

        image_token_len = 256

        if gradio_input:
            image = image_file.copy()
        else:
            image = self.load_image(image_file)

        w, h = image.size
        
        if ocr_type == 'format':
            qs = 'OCR with format: '
        else:
            qs = 'OCR: '

        if ocr_box:
            bbox = eval(ocr_box)
            if len(bbox) == 2:
                bbox[0] = int(bbox[0]/w*1000)
                bbox[1] = int(bbox[1]/h*1000)
            if len(bbox) == 4:
                bbox[0] = int(bbox[0]/w*1000)
                bbox[1] = int(bbox[1]/h*1000)
                bbox[2] = int(bbox[2]/w*1000)
                bbox[3] = int(bbox[3]/h*1000)
            if ocr_type == 'format':
                qs = str(bbox) + ' ' + 'OCR with format: '
            else:
                qs = str(bbox) + ' ' + 'OCR: '

        if ocr_color:
            if ocr_type == 'format':
                qs = '[' + ocr_color + ']' + ' ' + 'OCR with format: '
            else:
                qs = '[' + ocr_color + ']' + ' ' + 'OCR: '

        if use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs 
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs


        conv_mpt = Conversation(
            system="""<|im_start|>system
        You should follow the instructions carefully and explain your answers in detail.""",
            # system = None,
            roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
            version="mpt",
            messages=(),
            offset=0,
            sep_style=SeparatorStyle.MPT,
            sep="<|im_end|>",
        )

        conv = conv_mpt.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if print_prompt:
            print(prompt)

        inputs = tokenizer([prompt])

        image_tensor_1 = image_processor_high(image)

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        if stream_flag:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = self.generate(
                    input_ids,
                    images=[image_tensor_1.unsqueeze(0).half().cuda()],
                    do_sample=False,
                    num_beams = 1,
                    no_repeat_ngram_size = 20,
                    streamer=streamer,
                    max_new_tokens=4096,
                    stopping_criteria=[stopping_criteria]
                    )
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = self.generate(
                    input_ids,
                    images=[image_tensor_1.unsqueeze(0).half().cuda()],
                    do_sample=False,
                    num_beams = 1,
                    no_repeat_ngram_size = 20,
                    # streamer=streamer,
                    max_new_tokens=4096,
                    stopping_criteria=[stopping_criteria]
                    )
            
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        response_str = outputs

        return response_str

    def dynamic_preprocess(self, image, min_num=1, max_num=6, image_size=1024, use_thumbnail=True):
        
        def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
            best_ratio_diff = float('inf')
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
            return best_ratio
        
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        # print(target_ratios)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # print(target_aspect_ratio)
        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images


    def chat_crop(self, tokenizer, image_file, ocr_type, render=False, save_render_file=None, print_prompt=False, gradio_input=False, stream_flag = False):
        # Model
        self.disable_torch_init()
        multi_page=False


        image_processor_high =  GOTImageEvalProcessor(image_size=1024)

        use_im_start_end = True


        image_token_len = 256

        image_list = []

        # if len(image_file_list)>1:
        #     multi_page = True

        if multi_page:
            qs = 'OCR with format across multi pages: '
            # only for png files
            # import glob
            # from natsort import natsorted
            # patches = glob.glob(image_file + '/*png')
            patches = image_file
            # patches = natsorted(patches)
            sub_images = []
            for sub_image in patches:
                sub_images.append(self.load_image(sub_image))

            ll = len(patches)
            # print(patches)
            # print("len ll: ", ll)

        else:
            if ocr_type == 'format':
                qs = 'OCR with format upon the patch reference: '
            else:
                qs = 'OCR upon the patch reference: '
            if gradio_input:
                img = image_file.copy()
            else:
                img = self.load_image(image_file)
            sub_images = self.dynamic_preprocess(img)
            ll = len(sub_images)

        for image in sub_images:
            image_tensor_1 = image_processor_high(image)
            image_list.append(image_tensor_1)


        image_list = torch.stack(image_list)

        print('====new images batch size======:  \n',image_list.shape)


        if use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len*ll + DEFAULT_IM_END_TOKEN + '\n' + qs 
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs


        conv_mpt = Conversation(
            system="""<|im_start|>system
        You should follow the instructions carefully and explain your answers in detail.""",
            # system = None,
            roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
            version="mpt",
            messages=(),
            offset=0,
            sep_style=SeparatorStyle.MPT,
            sep="<|im_end|>",
        )

        conv = conv_mpt.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if print_prompt:
            print(prompt)

        inputs = tokenizer([prompt])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        if stream_flag:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = self.generate(
                    input_ids,
                    images=[image_list.half().cuda()],
                    do_sample=False,
                    num_beams = 1,
                    # no_repeat_ngram_size = 20,
                    streamer=streamer,
                    max_new_tokens=4096,
                    stopping_criteria=[stopping_criteria]
                    )
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = self.generate(
                    input_ids,
                    images=[image_list.half().cuda()],
                    do_sample=False,
                    num_beams = 1,
                    # no_repeat_ngram_size = 20,
                    # streamer=streamer,
                    max_new_tokens=4096,
                    stopping_criteria=[stopping_criteria]
                    )

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()   
        response_str = outputs

        return response_str

def main():
    with open('config/ocr/test.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['device'])
    print(f"Device used: {device}")

    model_name = config['model']['name'] #'allenai/Molmo-7B-D-0924'
    # from vllm import LLM

    # llm = LLM(model=model_name,tensor_parallel_size=2)  # Name or path of your model
    # # llm.todeivce(device)
    # prompt = "Output the structured output of OCR"
    # # Single prompt inference
    # outputs = llm.generate({
    #     "prompt": prompt,
    #     "multi_modal_data": {"image": image_file},
    # })

    # for o in outputs:
    #     generated_text = o.outputs[0].text
    #     print(generated_text)
    


    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        cache_dir=config['model']['cache_dir'], 
        trust_remote_code=True,
    )
    model = OcrHfDatasetsModel.from_pretrained(
        model_name, 
        cache_dir=config['model']['cache_dir'], 
        trust_remote_code=True, 
        low_cpu_mem_usage=True, 
        device_map=device, 
        use_safetensors=True, 
        pad_token_id=tokenizer.eos_token_id,
    )
    model = model.eval()
    model.to(device=device,  dtype=torch.bfloat16)

    # Load the entire dataset
    dataset = cast(Dataset, load_dataset(config['dataset']['input_path'], split=config['dataset']['split']))
    print("Dataset loaded: ", dataset)
    
    # # Set batch size
    # batch_size = 16

    # # Create a data loader for the dataset
    # image_data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=False,  # Set to True if you want to shuffle the data
    #     # num_workers=4,  # Number of worker threads for loading data
    #     pin_memory=True  # If you're using a GPU, this can speed up data transfer
    # )
    # Process the dataset in batches
    for idx, item in enumerate(tqdm(dataset, desc="Indexing Progress")):
        # Assuming the batch is a dictionary with 'image' as the key for images
        image = item['image']
        
        # Assuming the model has a method to process a batch of images
        # You would need to replace 'chat_crop' with the appropriate method for your model
        # res = model.chat_crop(tokenizer, images, ocr_type='format')
        res = model.chat(tokenizer, image, ocr_type='format')
        
        # render the formatted OCR results:
        # res = model.chat(tokenizer, images, ocr_type='format', render=True, save_render_file='./demo.html')
        # print("The result of OCR for batch: ")
        # print(res)

    # plain texts OCR
    # res = model.chat(tokenizer, image_file, ocr_type='ocr')

    # format texts OCR:
    # res = model.chat(tokenizer, image_file, ocr_type='format')

    # fine-grained OCR:
    # res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_box='')
    # res = model.chat(tokenizer, image_file, ocr_type='format', ocr_box='')
    # res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_color='')
    # res = model.chat(tokenizer, image_file, ocr_type='format', ocr_color='')

    # multi-crop OCR:
    # res = model.chat_crop(tokenizer, image_file, ocr_type='ocr')
    # res = model.chat_crop(tokenizer, image_file, ocr_type='format')

    # render the formatted OCR results:
    # res = model.chat(tokenizer, image_file, ocr_type='format', render=True, save_render_file = './demo.html')
    print("The result of OCR: ")
    print(res)

if __name__ == '__main__':
    main()