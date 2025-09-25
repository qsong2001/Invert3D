# Invert3D: Align 3D Representation and Text Embedding for 3D Content Personalization \[[arxiv](https://arxiv.org/abs/2508.16932)\]


## Description
This repo contains the official code, data and sample inversions for our 3D-to-text inversion paper. 

**25/9/2025** Code released!


# Setup
```
git clone https://github.com/qsong2001/Invert3D.git
```
Our code builds on, and shares requirements with MVDream and Textural inversion. To set up their environment, please run:
```
cd textual_inversion
conda env create -f environment.yaml
conda activate text

cd ..
pip install MVDream -v -e .
```

To render novel view from Gaussian splatting, you need to install  diff_gaussian_rasterization
```
pip install -e git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git@main#egg=diff_gaussian_rasterization
pip install -e git+https://github.com/camenduru/simple-knn@main#egg=simple_knn
```


You will also need the official MVDream text-to-3D checkpoint (sd-v1.5-4view), available through the [MVDream](https://github.com/bytedance/MVDream). 

The final organization of the folder should be:
```
├── input  # Demo data sample
├── MVDream
├── textual_inversion
└── sd-v1.5-4view.pt # text-to-3D pre-trained model
```

Due to the version problem of pytorch_lighting, you shoud comment the code in lines 312-316 of src/taming-transformers/main.py.
```
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")
```



# Inversion

To invert a 3D GS sample, run:
```
python main.py --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml 
               -t 
               --actual_resume /path/to/pretrained/model.ckpt 
               -n <run_name> 
               --gpus 0, 
               --data_root /path/to/directory/with/images
               --init_word <initialization_word>
```

For example:
```
python inverse_mvdream.py --base configs/mvdream/real-mvdream-v1-finetune-obj-v32-any.yaml -t --actual_resume ../sd-v1.5-4view.pt -n test-mv-base-model-32-vector-any-view --gpus 0, --data_root ../input/demo/3D-GS/a_pikachu_with_smily_face.ply
```
The final results are save in folder: ./logs.

We are very appreciate for the open-source projects: [MV-Dream](https://github.com/bytedance/MVDream) and [textural_inversion](https://github.com/rinongal/textual_inversion).





