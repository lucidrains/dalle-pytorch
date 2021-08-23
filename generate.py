import argparse
from pathlib import Path
from tqdm import tqdm

# torch

import torch

from einops import repeat

# vision imports

from PIL import Image
from torchvision.utils import make_grid, save_image

# dalle related classes and utils

from dalle_pytorch import DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE, DALLE
from dalle_pytorch.tokenizer import tokenizer, HugTokenizer, YttmTokenizer, ChineseTokenizer

# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument('--dalle_path', type = str, required = True,
                    help='path to your trained DALL-E')

parser.add_argument('--vqgan_model_path', type=str, default = None,
                   help='path to your trained VQGAN weights. This should be a .ckpt file. (only valid when taming option is enabled)')

parser.add_argument('--vqgan_config_path', type=str, default = None,
                   help='path to your trained VQGAN config. This should be a .yaml file.  (only valid when taming option is enabled)')

parser.add_argument('--text', type = str, required = True,
                    help='your text prompt')

parser.add_argument('--num_images', type = int, default = 128, required = False,
                    help='number of images')

parser.add_argument('--batch_size', type = int, default = 4, required = False,
                    help='batch size')

parser.add_argument('--top_k', type = float, default = 0.9, required = False,
                    help='top k filter threshold')

parser.add_argument('--image_size', type = int, default = 256, required = False,
                    help='image size')

parser.add_argument('--outputs_dir', type = str, default = './outputs', required = False,
                    help='output directory')

parser.add_argument('--bpe_path', type = str,
                    help='path to your huggingface BPE json file')

parser.add_argument('--hug', dest='hug', action = 'store_true')

parser.add_argument('--chinese', dest='chinese', action = 'store_true')

parser.add_argument('--taming', dest='taming', action='store_true')

parser.add_argument('--gentxt', dest='gentxt', action='store_true')

args = parser.parse_args()

# helper fns

def exists(val):
    return val is not None

# tokenizer

if exists(args.bpe_path):
    klass = HugTokenizer if args.hug else YttmTokenizer
    tokenizer = klass(args.bpe_path)
elif args.chinese:
    tokenizer = ChineseTokenizer()

# load DALL-E

dalle_path = Path(args.dalle_path)

assert dalle_path.exists(), 'trained DALL-E must exist'

load_obj = torch.load(str(dalle_path))
dalle_params, vae_params, weights = load_obj.pop('hparams'), load_obj.pop('vae_params'), load_obj.pop('weights')

dalle_params.pop('vae', None) # cleanup later

IMAGE_SIZE = args.image_size

if vae_params is not None:
    vae = DiscreteVAE(IMAGE_SIZE, **vae_params[1:])
elif not args.taming:
    vae = OpenAIDiscreteVAE(IMAGE_SIZE)
else:
    vae = VQGanVAE(IMAGE_SIZE, args.vqgan_model_path, args.vqgan_config_path)


dalle = DALLE(vae = vae, **dalle_params).cuda()

dalle.load_state_dict(weights)

# generate images

texts = args.text.split('|')

for j, text in tqdm(enumerate(texts)):
    if args.gentxt:
        text_tokens, gen_texts = dalle.generate_texts(tokenizer, text=text, filter_thres = args.top_k)
        text = gen_texts[0]
    else:
        text_tokens = tokenizer.tokenize([text], dalle.text_seq_len).cuda()

    text_tokens = repeat(text_tokens, '() n -> b n', b = args.num_images)

    outputs = []

    for text_chunk in tqdm(text_tokens.split(args.batch_size), desc = f'generating images for - {text}'):
        output = dalle.generate_images(text_chunk, filter_thres = args.top_k)
        outputs.append(output)

    outputs = torch.cat(outputs)

    # save all images
    file_name = text 
    outputs_dir = Path(args.outputs_dir) / file_name.replace(' ', '_')[:(100)]
    outputs_dir.mkdir(parents = True, exist_ok = True)

    for i, image in tqdm(enumerate(outputs), desc = 'saving images'):
        save_image(image, outputs_dir / f'{i}.jpg', normalize=True)
        with open(outputs_dir / 'caption.txt', 'w') as f:
            f.write(file_name)

    print(f'created {args.num_images} images at "{str(outputs_dir)}"')
