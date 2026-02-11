# MR-CLIP

Codes for the paper of "Evidence-grounded brain MRI reporting via expert-foundation models cooperation".

This module implements one of the implementations of MR-CLIP. It uses MR-ViT as the image encoder, which is a multi-sequence image input framework based on CT-ViT, and uses a pre-trained medical language model as the text encoder.

## Installation

To install MR-CLIP, execute the following command:

```bash
$ pip install -e .
```

This will install the necessary dependencies and enable you to use MR-CLIP.

## Usage

```python
from mr_clip.transformer_maskgit import MRViT
from mr_clip.transformers import BertTokenizer, BertModel
from mr_clip import MRCLIP
from pathlib import Path
from os.path import join


def init_model(model_root, cfg, device):
    # Initialize BERT tokenizer and text encoder
    LM_PATH = join(model_root, cfg.bert_pretrained)
    tokenizer = BertTokenizer.from_pretrained(LM_PATH, do_lower_case=True)
    text_encoder = BertModel.from_pretrained(LM_PATH)
    text_encoder.resize_token_embeddings(len(tokenizer))
    text_encoder = text_encoder.to(device)

    # Initialize image encoder and clip model
    image_encoder = MRViT(
        dim=512,
        codebook_size=8192,
        image_size=256,
        patch_size=32,
        temporal_patch_size=6,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
        channels=cfg.modal_channels)
    # 将图像编码器移动到指定GPU
    image_encoder = image_encoder.to(device)

    clip = MRCLIP(
        image_encoder=image_encoder,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        dim_image=262144,    # 8*8*8*512 (48/6, 256/32)
        # dim_text=768,
        dim_text=1024,  # chinese bert
        dim_latent=512,
        # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False)

    clip = clip.to(device)
    clip.load(Path(model_root)/cfg.clip_pretrained, map_location=device)
    # 确保模型在正确的设备上
    clip = clip.to(device)

    return tokenizer, clip

```
