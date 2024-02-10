import os
from cog import BasePredictor, Input, Path, BaseModel
from pyngrok import ngrok, conf

class Output(BaseModel):
    path: Path
    text: str

from PIL import Image
import numpy as np
import torch as T
import transformers, diffusers
from llava.conversation import conv_templates
from llava.model import *

DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
DEFAULT_IM_START_TOKEN = '<im_start>'
DEFAULT_IM_END_TOKEN = '<im_end>'
PATH_LLAVA = '/content/ml-mgie-hf/data/LLaVA-7B-v1'

def crop_resize(f, sz=512):
    w, h = f.size
    if w>h:
        p = (w-h)//2
        f = f.crop([p, 0, p+h, h])
    elif h>w:
        p = (h-w)//2
        f = f.crop([0, p, w, p+w])
    f = f.resize([sz, sz])
    return f

def remove_alter(s):  # hack expressive instruction
    if 'ASSISTANT:' in s: s = s[s.index('ASSISTANT:')+10:].strip()
    if '</s>' in s: s = s[:s.index('</s>')].strip()
    if 'alternative' in s.lower(): s = s[:s.lower().index('alternative')]
    if '[IMG0]' in s: s = s[:s.index('[IMG0]')]
    s = '.'.join([s.strip() for s in s.split('.')[:2]])
    if s[-1]!='.': s += '.'
    return s.strip()

def go_mgie(img, txt, seed, cfg_txt, cfg_img, image_processor, image_token_len, EMB, tokenizer, model, pipe):
    img, seed = crop_resize(Image.fromarray(img).convert('RGB')), int(seed)
    inp = img

    img = image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
    txt = "what will this image be like if '%s'"%(txt)
    txt = txt+'\n'+DEFAULT_IM_START_TOKEN+DEFAULT_IMAGE_PATCH_TOKEN*image_token_len+DEFAULT_IM_END_TOKEN
    conv = conv_templates['vicuna_v1_1'].copy()
    conv.append_message(conv.roles[0], txt), conv.append_message(conv.roles[1], None)
    txt = conv.get_prompt()
    txt = tokenizer(txt)
    txt, mask = T.as_tensor(txt['input_ids']), T.as_tensor(txt['attention_mask'])

    with T.inference_mode():
        out = model.generate(txt.unsqueeze(dim=0).cuda(), images=img.half().unsqueeze(dim=0).cuda(), attention_mask=mask.unsqueeze(dim=0).cuda(), 
                             do_sample=False, max_new_tokens=96, num_beams=1, no_repeat_ngram_size=3, 
                             return_dict_in_generate=True, output_hidden_states=True)
        out, hid = out['sequences'][0].tolist(), T.cat([x[-1] for x in out['hidden_states']], dim=1)[0]
        
        if 32003 in out: p = out.index(32003)-1
        else: p = len(hid)-9
        p = min(p, len(hid)-9)
        hid = hid[p:p+8]

        out = remove_alter(tokenizer.decode(out))
        emb = model.edit_head(hid.unsqueeze(dim=0), EMB)
        res = pipe(image=inp, prompt_embeds=emb, negative_prompt_embeds=None, 
                   generator=T.Generator(device='cuda').manual_seed(seed), guidance_scale=cfg_txt, image_guidance_scale=cfg_img).images[0]
        res.save("/content/image.png")
    return "/content/image.png", out
    
class Predictor(BasePredictor):
    def setup(self) -> None:
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(PATH_LLAVA)
        self.model = LlavaLlamaForCausalLM.from_pretrained(PATH_LLAVA, low_cpu_mem_usage=True, torch_dtype=T.float16, use_cache=True).cuda()
        self.image_processor = transformers.CLIPImageProcessor.from_pretrained(self.model.config.mm_vision_tower, torch_dtype=T.float16)
        
        self.tokenizer.padding_side = 'left'
        self.tokenizer.add_tokens(['[IMG0]', '[IMG1]', '[IMG2]', '[IMG3]', '[IMG4]', '[IMG5]', '[IMG6]', '[IMG7]'], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))
        ckpt = T.load('/content/ml-mgie-hf/data/mgie_7b/mllm.pt', map_location='cpu')
        self.model.load_state_dict(ckpt, strict=False)
        
        mm_use_im_start_end = getattr(self.model.config, 'mm_use_im_start_end', False)
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end: self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        
        vision_tower = self.model.get_model().vision_tower[0]
        vision_tower = transformers.CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=T.float16, low_cpu_mem_usage=True).cuda()
        self.model.get_model().vision_tower[0] = vision_tower
        vision_config = vision_tower.config
        vision_config.im_patch_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end: vision_config.im_start_token, vision_config.im_end_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        self.image_token_len = (vision_config.image_size//vision_config.patch_size)**2
        
        _ = self.model.eval()
        self.EMB = ckpt['emb'].cuda()
        with T.inference_mode(): NULL = self.model.edit_head(T.zeros(1, 8, 4096).half().to('cuda'), self.EMB)
        
        self.pipe = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained('timbrooks/instruct-pix2pix', torch_dtype=T.float16).to('cuda')
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.unet.load_state_dict(T.load('/content/ml-mgie-hf/data/mgie_7b/unet.pt', map_location='cpu'))
        print('--init MGIE--')
    def predict(
        self,
        input_image: Path = Input(description="Input Image"),
        prompt: str = Input(default="make the frame red"),
        seed: int = Input(default=13331),
        text_cfg: float = Input(default=7.5),
        image_cfg: float = Input(default=1.5),
    ) -> Output:
        res, out = go_mgie(np.array(Image.open(input_image).convert('RGB')), prompt, seed, text_cfg, image_cfg, self.image_processor, self.image_token_len, self.EMB, self.tokenizer, self.model, self.pipe)
        return Output(path = Path(res), text = out)