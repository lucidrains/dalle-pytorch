import argparse
from pathlib import Path
import time
from glob import glob
import os
import shutil

import torch
import wandb  # Quit early if user doesn't have wandb installed.
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE, DiscreteVAE, DALLE
from dalle_pytorch import distributed_utils
from dalle_pytorch.loader import TextImageDataset
from dalle_pytorch.tokenizer import tokenizer, HugTokenizer, ChineseTokenizer, YttmTokenizer

# libraries needed for webdataset support
import webdataset as wds
from torchvision import transforms as T
from PIL import Image
from io import BytesIO


# argument parsing

parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required=False)

group.add_argument('--vae_path', type=str,
                   help='path to your trained discrete VAE')

group.add_argument('--dalle_path', type=str,
                   help='path to your partially trained DALL-E')

parser.add_argument('--vqgan_model_path', type=str, default = None,
                   help='path to your trained VQGAN weights. This should be a .ckpt file. (only valid when taming option is enabled)')

parser.add_argument('--vqgan_config_path', type=str, default = None,
                   help='path to your trained VQGAN config. This should be a .yaml file. (only valid when taming option is enabled)')

parser.add_argument('--image_text_folder', type=str, required=True,
                    help='path to your folder of images and text for learning the DALL-E')

parser.add_argument(
    '--wds', 
    type = str, 
    default='', 
    help = 'Comma separated list of WebDataset (1) image and (2) text column names. Must contain 2 values, e.g. img,cap.'
)

parser.add_argument('--truncate_captions', dest='truncate_captions', action='store_true',
                    help='Captions passed in which exceed the max token length will be truncated if this is set.')

parser.add_argument('--random_resize_crop_lower_ratio', dest='resize_ratio', type=float, default=0.75,
                    help='Random resized crop lower ratio')

parser.add_argument('--chinese', dest='chinese', action='store_true')

parser.add_argument('--taming', dest='taming', action='store_true')

parser.add_argument('--hug', dest='hug', action='store_true')

parser.add_argument('--bpe_path', type=str,
                    help='path to your BPE json file')

parser.add_argument('--dalle_output_file_name', type=str, default = "dalle",
                    help='output_file_name')

parser.add_argument('--fp16', action='store_true',
                    help='(experimental) - Enable DeepSpeed 16 bit precision. Reduces VRAM.')


parser.add_argument('--amp', action='store_true',
	help='Apex "O1" automatic mixed precision. More stable than 16 bit precision. Can\'t be used in conjunction with deepspeed zero stages 1-3.')

parser.add_argument('--wandb_name', default='dalle_train_transformer',
                    help='Name W&B will use when saving results.\ne.g. `--wandb_name "coco2017-full-sparse"`')

parser.add_argument('--wandb_entity', default=None,
                    help='(optional) Name of W&B team/entity to log to.')

parser.add_argument('--stable_softmax', dest='stable_softmax', action='store_true',
                    help='Prevent values from becoming too large during softmax. Helps with stability in fp16 and Mixture of Quantization training.')

parser = distributed_utils.wrap_arg_parser(parser)

train_group = parser.add_argument_group('Training settings')

train_group.add_argument('--flops_profiler', dest = 'flops_profiler', action='store_true', help = 'Exits after printing detailed flops/runtime analysis of forward/backward')

train_group.add_argument('--epochs', default = 20, type = int, help = 'Number of epochs')

train_group.add_argument('--save_every_n_steps', default = 1000, type = int, help = 'Save a checkpoint every n steps')

train_group.add_argument('--keep_n_checkpoints', default = None, type = int, help = '(Careful) Deletes old deepspeed checkpoints if there are more than n')

train_group.add_argument('--batch_size', default = 4, type = int, help = 'Batch size')

train_group.add_argument('--ga_steps', default = 1, type = int, help = 'Number of steps to accumulate gradients across per each iteration. DeepSpeed only.')

train_group.add_argument('--learning_rate', default = 3e-4, type = float, help = 'Learning rate')

train_group.add_argument('--clip_grad_norm', default = 0.5, type = float, help = 'Clip gradient norm')

train_group.add_argument('--lr_decay', dest = 'lr_decay', action = 'store_true')

model_group = parser.add_argument_group('Model settings')

model_group.add_argument('--dim', default = 512, type = int, help = 'Model dimension')

model_group.add_argument('--text_seq_len', default = 256, type = int, help = 'Text sequence length')

model_group.add_argument('--depth', default = 2, type = int, help = 'Model depth')

model_group.add_argument('--heads', default = 8, type = int, help = 'Model number of heads')

model_group.add_argument('--dim_head', default = 64, type = int, help = 'Model head dimension')

train_group.add_argument('--ff_dropout', default = 0.0, type = float, help = 'Feed forward dropout.')

train_group.add_argument('--attn_dropout', default = 0.0, type = float, help = 'Feed forward dropout.')

model_group.add_argument('--reversible', dest = 'reversible', action='store_true')

model_group.add_argument('--loss_img_weight', default = 7, type = int, help = 'Image loss weight')

model_group.add_argument('--image_size', default = 256, type = int, help = 'Image size')

model_group.add_argument('--attn_types', default = 'full', type = str, help = 'comma separated list of attention types. attention type can be: full or sparse or axial_row or axial_col or conv_like.')

model_group.add_argument('--shift_tokens', help = 'Use the shift tokens feature', action = 'store_true')

model_group.add_argument('--rotary_emb', help = 'Use rotary embeddings', action = 'store_true')

args = parser.parse_args()

# helpers

def exists(val):
    return val is not None

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]

def cp_path_to_dir(cp_path, tag):
    """Convert a checkpoint path to a directory with `tag` inserted.
    If `cp_path` is already a directory, return it unchanged.
    """
    if not isinstance(cp_path, Path):
        cp_path = Path(cp_path)
    if cp_path.is_dir():
        return cp_path
    path_sans_extension = cp_path.parent / cp_path.stem
    cp_dir = Path(f'{path_sans_extension}-{tag}-cp')
    return cp_dir

# constants
WEBDATASET_IMAGE_TEXT_COLUMNS = tuple(args.wds.split(','))
ENABLE_WEBDATASET = True if len(WEBDATASET_IMAGE_TEXT_COLUMNS) == 2 else False

DALLE_OUTPUT_FILE_NAME = args.dalle_output_file_name + ".pt"

VAE_PATH = args.vae_path
VQGAN_MODEL_PATH = args.vqgan_model_path
VQGAN_CONFIG_PATH = args.vqgan_config_path
DALLE_PATH = args.dalle_path
RESUME = exists(DALLE_PATH)

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

LEARNING_RATE = args.learning_rate
GRAD_CLIP_NORM = args.clip_grad_norm
LR_DECAY = args.lr_decay
SAVE_EVERY_N_STEPS = args.save_every_n_steps
KEEP_N_CHECKPOINTS = args.keep_n_checkpoints

IMAGE_SIZE = args.image_size
MODEL_DIM = args.dim
TEXT_SEQ_LEN = args.text_seq_len
DEPTH = args.depth
HEADS = args.heads
DIM_HEAD = args.dim_head
REVERSIBLE = args.reversible
LOSS_IMG_WEIGHT = args.loss_img_weight
FF_DROPOUT = args.ff_dropout
ATTN_DROPOUT = args.attn_dropout
STABLE = args.stable_softmax
SHIFT_TOKENS = args.shift_tokens
ROTARY_EMB = args.rotary_emb

ATTN_TYPES = tuple(args.attn_types.split(','))

DEEPSPEED_CP_AUX_FILENAME = 'auxiliary.pt'

if not ENABLE_WEBDATASET:
    # quit early if you used the wrong folder name
    assert Path(args.image_text_folder).exists(), f'The path {args.image_text_folder} was not found.'
else:
    # quit early if no tar files were found
    if Path(args.image_text_folder).is_dir():
        DATASET = [str(p) for p in Path(args.image_text_folder).glob("**/*") if ".tar" in str(p).lower()] # .name
        assert len(DATASET) > 0, 'The directory ({}) does not contain any WebDataset/.tar files.'.format(args.image_text_folder)
        print('Found {} WebDataset .tar(.gz) file(s) under given path {}!'.format(len(DATASET), args.image_text_folder))
    elif ('http://' in args.image_text_folder.lower()) | ('https://' in args.image_text_folder.lower()):
        DATASET = f"pipe:curl -L -s {args.image_text_folder} || true"
        print('Found {} http(s) link under given path!'.format(len(DATASET), args.image_text_folder))
    elif 'gs://' in args.image_text_folder.lower():
        DATASET = f"pipe:gsutil cat {args.image_text_folder} || true"
        print('Found {} GCS link under given path!'.format(len(DATASET), args.image_text_folder))
    elif '.tar' in args.image_text_folder:
        DATASET = args.image_text_folder
        print('Found WebDataset .tar(.gz) file under given path {}!'.format(args.image_text_folder))
    else:
        raise Exception('No folder, no .tar(.gz) and no url pointing to tar files provided under {}.'.format(args.image_text_folder))

# initialize distributed backend

distr_backend = distributed_utils.set_backend_from_args(args)
distr_backend.initialize()

using_deepspeed = \
    distributed_utils.using_backend(distributed_utils.DeepSpeedBackend)

# tokenizer

if exists(args.bpe_path):
    klass = HugTokenizer if args.hug else YttmTokenizer
    tokenizer = klass(args.bpe_path)
elif args.chinese:
    tokenizer = ChineseTokenizer()

# reconstitute vae
if RESUME:
    dalle_path = Path(DALLE_PATH)
    if using_deepspeed:
        cp_dir = cp_path_to_dir(dalle_path, 'ds')
        assert cp_dir.is_dir(), \
            f'DeepSpeed checkpoint directory {cp_dir} not found'
        dalle_path = cp_dir / DEEPSPEED_CP_AUX_FILENAME
    else:
        assert dalle_path.exists(), 'DALL-E model file does not exist'
    loaded_obj = torch.load(str(dalle_path), map_location='cpu')

    dalle_params, vae_params, weights = loaded_obj['hparams'], loaded_obj['vae_params'], loaded_obj['weights']
    opt_state = loaded_obj.get('opt_state')
    scheduler_state = loaded_obj.get('scheduler_state')

    if vae_params is not None:
        vae = DiscreteVAE(IMAGE_SIZE, **vae_params[1:])
    else:
        if args.taming:
            vae = VQGanVAE(IMAGE_SIZE, VQGAN_MODEL_PATH, VQGAN_CONFIG_PATH)
        else:
            vae = OpenAIDiscreteVAE(IMAGE_SIZE)

    dalle_params = dict(
        **dalle_params
    )
    resume_epoch = loaded_obj.get('epoch', 0)
else:
    if exists(VAE_PATH):
        vae_path = Path(VAE_PATH)
        assert vae_path.exists(), 'VAE model file does not exist'
        assert not vae_path.is_dir(), \
            ('Cannot load VAE model from directory; please use a '
             'standard *.pt checkpoint. '
             'Currently, merging a DeepSpeed-partitioned VAE into a DALLE '
             'model is not supported.')

        loaded_obj = torch.load(str(vae_path))

        vae_params, weights = loaded_obj['hparams'], loaded_obj['weights']

        vae = DiscreteVAE(IMAGE_SIZE, **vae_params[1:])
        vae.load_state_dict(weights)
    else:
        if distr_backend.is_root_worker():
            print('using pretrained VAE for encoding images to tokens')
        vae_params = None

        if args.taming:
            vae = VQGanVAE(IMAGE_SIZE, VQGAN_MODEL_PATH, VQGAN_CONFIG_PATH)
        else:
            vae = OpenAIDiscreteVAE(IMAGE_SIZE)

    dalle_params = dict(
        num_text_tokens=tokenizer.vocab_size,
        text_seq_len=TEXT_SEQ_LEN,
        dim=MODEL_DIM,
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD,
        reversible=REVERSIBLE,
        loss_img_weight=LOSS_IMG_WEIGHT,
        attn_types=ATTN_TYPES,
        ff_dropout=FF_DROPOUT,
        attn_dropout=ATTN_DROPOUT,
        stable=STABLE,
        shift_tokens=SHIFT_TOKENS,
        rotary_emb=ROTARY_EMB,
    )
    resume_epoch = 0

# configure OpenAI VAE for float16s

if isinstance(vae, OpenAIDiscreteVAE) and args.fp16:
    vae.enc.blocks.output.conv.use_float16 = True


# helpers

def group_weight(model):
    group_decay, group_no_decay = [], []
    for params in model.named_parameters():
        if 'transformer' in params[0]:
            if 'bias' in params[0] or 'norm' in params[0]:
                group_no_decay.append(params[1])
                continue
        group_decay.append(params[1])

    assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


# create dataset and dataloader

is_shuffle = not distributed_utils.using_backend(distributed_utils.HorovodBackend)

imagepreproc = T.Compose([
    T.Lambda(lambda img: img.convert('RGB')
    if img.mode != 'RGB' else img),
    T.RandomResizedCrop(IMAGE_SIZE,
                        scale=(args.resize_ratio, 1.),
                        ratio=(1., 1.)),
    T.ToTensor(),
])

def imagetransform(b):
    return Image.open(BytesIO(b))

def tokenize(s):
    return tokenizer.tokenize(
        s.decode('utf-8'),
        TEXT_SEQ_LEN,
        truncate_text=args.truncate_captions).squeeze(0)

if ENABLE_WEBDATASET:
    DATASET_SIZE = int(1e9) # You need to set a nominal length for the Dataset in order to avoid warnings from DataLoader

    myimg, mycap = WEBDATASET_IMAGE_TEXT_COLUMNS
    image_text_mapping = {
        myimg: imagetransform,
        mycap: tokenize
    }
    image_mapping = {
        myimg: imagepreproc
    }

    num_batches = DATASET_SIZE // BATCH_SIZE

    ds = (
        wds.WebDataset(DATASET, length=num_batches)
        # .shuffle(is_shuffle) # Commented out for WebDataset as the behaviour cannot be predicted yet
        .map_dict(**image_text_mapping)     
        .map_dict(**image_mapping)
        .to_tuple(mycap, myimg)
        .batched(BATCH_SIZE, partial=False) # It is good to avoid partial batches when using Distributed training
    )
else:
    ds = TextImageDataset(
        args.image_text_folder,
        text_len=TEXT_SEQ_LEN,
        image_size=IMAGE_SIZE,
        resize_ratio=args.resize_ratio,
        truncate_captions=args.truncate_captions,
        tokenizer=tokenizer,
        shuffle=is_shuffle,
    )

assert len(ds) > 0, 'dataset is empty'
if distr_backend.is_root_worker():
    if not ENABLE_WEBDATASET:
        print(f'{len(ds)} image-text pairs found for training')

if not is_shuffle:
    data_sampler = torch.utils.data.distributed.DistributedSampler(
        ds,
        num_replicas=distr_backend.get_world_size(),
        rank=distr_backend.get_rank()
    )
else:
    data_sampler = None

if ENABLE_WEBDATASET:
    # WebLoader for WebDataset and DeepSpeed compatibility
    dl = wds.WebLoader(ds, batch_size=None, shuffle=False) # optionally add num_workers=2 (n) argument
    number_of_batches = DATASET_SIZE // (BATCH_SIZE * distr_backend.get_world_size())
    dl = dl.repeat(2).slice(number_of_batches)
    dl.length = number_of_batches
else:
    # Regular DataLoader for image-text-folder datasets
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=is_shuffle, drop_last=True, sampler=data_sampler)


# initialize DALL-E

dalle = DALLE(vae=vae, **dalle_params)
if not using_deepspeed:
    if args.fp16:
        dalle = dalle.half()
    dalle = dalle.cuda()

if RESUME and not using_deepspeed:
    dalle.load_state_dict(weights)

# optimizer

opt = Adam(get_trainable_params(dalle), lr=LEARNING_RATE)
if RESUME and opt_state:
    opt.load_state_dict(opt_state)

if LR_DECAY:
    scheduler = ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=10,
        cooldown=10,
        min_lr=1e-6,
        verbose=True,
    )
    if RESUME and scheduler_state:
        scheduler.load_state_dict(scheduler_state)
else:
    scheduler = None

if distr_backend.is_root_worker():
    # experiment tracker

    model_config = dict(
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD
    )

    run = wandb.init(
        project=args.wandb_name,
        entity=args.wandb_entity,
        resume=False,
        config=model_config,
    )

# distribute

distr_backend.check_batch_size(BATCH_SIZE)
deepspeed_config = {
    'train_batch_size': BATCH_SIZE,
    'gradient_accumulation_steps': args.ga_steps,
    'gradient_clipping': GRAD_CLIP_NORM,
    'fp16': {
        'enabled': args.fp16,
    },
    'amp': {
        'enabled': args.amp,
        'opt_level': 'O1',
    },
    "flops_profiler": {
        "enabled": args.flops_profiler,
        "profile_step": 200,
        "module_depth": -1,
        "top_modules": 1,
        "detailed": True,
        "output_file": None # TODO Can't get this to work.
    },
}

if deepspeed_config.get('zero_optimization', {}).get('stage', 0) >= 2:
    print(f"Checkpoints made with DeepSpeed ZeRO Stages 2 and 3 will be stored in deepspeed checkpoint folder")
    print(f"As such, they will require DeepSpeed as a dependency in order to resume from or generate with.")
    print("See the deespeed conversion script for details on how to convert your ZeRO stage 2/3 checkpoint to a single file.")
    print("If using a single GPU, consider running with apex automatic mixed precision instead for a similar speedup to ZeRO.")
    time.sleep(2)

(distr_dalle, distr_opt, distr_dl, distr_scheduler) = distr_backend.distribute(
    args=args,
    model=dalle,
    optimizer=opt,
    model_parameters=get_trainable_params(dalle),
    training_data=ds if using_deepspeed else dl,
    # Do not pass the LR scheduler to DeepSpeed so we can manually
    # advance it.
    lr_scheduler=scheduler if LR_DECAY and not using_deepspeed else None,
    config_params=deepspeed_config,
)
# Prefer scheduler in `deepspeed_config`.
if LR_DECAY and distr_scheduler is None:
    distr_scheduler = scheduler
avoid_model_calls = using_deepspeed and args.fp16

if RESUME and using_deepspeed:
    distr_dalle.load_checkpoint(str(cp_dir))


def save_model(path, epoch=0):
    save_obj = {
        'hparams': dalle_params,
        'vae_params': vae_params,
        'epoch': epoch,
    }
    if using_deepspeed:
        cp_dir = cp_path_to_dir(path, 'ds')

        if KEEP_N_CHECKPOINTS is not None and distr_backend.is_root_worker():
            checkpoints = sorted(glob(str(cp_dir / "global*")), key=os.path.getmtime, reverse=True)
            for checkpoint in checkpoints[KEEP_N_CHECKPOINTS:]:
                shutil.rmtree(checkpoint)

        distr_dalle.save_checkpoint(cp_dir, client_state=save_obj)

        if not distr_backend.is_root_worker():
            return

        # Save auxiliary values so we can reuse the standard routine
        # for loading.
        save_obj = {
            **save_obj,
            # Save a nonsense value that directs the user to
            # further help.
            'weights': (
                'To get a working standard checkpoint, '
                'look into consolidating DeepSpeed checkpoints.'
            ),
        }
        torch.save(save_obj, str(cp_dir / DEEPSPEED_CP_AUX_FILENAME))
        if deepspeed_config.get('zero_optimization', {}).get('stage', 0) >= 2: # see https://github.com/lucidrains/DALLE-pytorch/wiki/DeepSpeed-Checkpoints
            return

    if not distr_backend.is_root_worker():
        return

    save_obj = {
        **save_obj,
        'weights': dalle.state_dict(),
        'opt_state': opt.state_dict(),
    }
    save_obj['scheduler_state'] = (scheduler.state_dict() if scheduler else None)
    torch.save(save_obj, path)

# training

# Saves a checkpoint before training begins to fail early when mis-configured.
# See https://github.com/lucidrains/DALLE-pytorch/wiki/DeepSpeed-Checkpoints
save_model(DALLE_OUTPUT_FILE_NAME, epoch=resume_epoch)
for epoch in range(resume_epoch, EPOCHS):
    if data_sampler:
        data_sampler.set_epoch(epoch)
    for i, (text, images) in enumerate((dl if ENABLE_WEBDATASET else distr_dl)):
        if i % 10 == 0 and distr_backend.is_root_worker():
            t = time.time()
        if args.fp16:
            images = images.half()
        text, images = map(lambda t: t.cuda(), (text, images))

        loss = distr_dalle(text, images, return_loss=True)

        if using_deepspeed:
            distr_dalle.backward(loss)
            distr_dalle.step()
            # Gradients are automatically zeroed after the step
        else:
            loss.backward()
            clip_grad_norm_(distr_dalle.parameters(), GRAD_CLIP_NORM)
            distr_opt.step()
            distr_opt.zero_grad()

        # Collective loss, averaged
        avg_loss = distr_backend.average_all(loss)

        log = {}

        if i % 10 == 0 and distr_backend.is_root_worker():
            print(epoch, i, f'loss - {avg_loss.item()}')

            log = {
                **log,
                'epoch': epoch,
                'iter': i,
                'loss': avg_loss.item()
            }

        if i % SAVE_EVERY_N_STEPS == 0:
            save_model(DALLE_OUTPUT_FILE_NAME, epoch=epoch)
	
        if i % 100 == 0:
            if distr_backend.is_root_worker():
                sample_text = text[:1]
                token_list = sample_text.masked_select(sample_text != 0).tolist()
                decoded_text = tokenizer.decode(token_list)

                if not avoid_model_calls:
                    # CUDA index errors when we don't guard this
                    image = dalle.generate_images(text[:1], filter_thres=0.9)  # topk sampling at 0.9


                log = {
                    **log,
                }
                if not avoid_model_calls:
                    log['image'] = wandb.Image(image, caption=decoded_text)

        if i % 10 == 9 and distr_backend.is_root_worker():
            sample_per_sec = BATCH_SIZE * 10 / (time.time() - t)
            log["sample_per_sec"] = sample_per_sec
            print(epoch, i, f'sample_per_sec - {sample_per_sec}')

        if i == 201 and args.flops_profiler:
            raise StopIteration("Profiler has finished running. Stopping training early.")

        if distr_backend.is_root_worker():
            wandb.log(log)

    if LR_DECAY:
        distr_scheduler.step(avg_loss)

    save_model(DALLE_OUTPUT_FILE_NAME, epoch=epoch)
    
    if distr_backend.is_root_worker():
        # save trained model to wandb as an artifact every epoch's end

        model_artifact = wandb.Artifact('trained-dalle', type='model', metadata=dict(model_config))
        model_artifact.add_file(DALLE_OUTPUT_FILE_NAME)
        run.log_artifact(model_artifact)

save_model(DALLE_OUTPUT_FILE_NAME, epoch=epoch)
if distr_backend.is_root_worker():
    wandb.save(DALLE_OUTPUT_FILE_NAME)
    model_artifact = wandb.Artifact('trained-dalle', type='model', metadata=dict(model_config))
    model_artifact.add_file(DALLE_OUTPUT_FILE_NAME)
    run.log_artifact(model_artifact)

    wandb.finish()
