# DA Wand [[Project Page](https://threedle.github.io/DA-Wand/)]
[![arXiv](https://img.shields.io/badge/arXiv-DAWand-b31b1b.svg)](https://arxiv.org/abs/2112.03221)
![Pytorch](https://img.shields.io/badge/PyTorch->=1.11.0-Red?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA->=11.3.1-Red?logo=CUDA)


Public code release for "DA Wand: Distortion-Aware Selection using Neural Mesh Parameterization".

## Getting Started
### Installation

```
conda env create --file dawand.yml
conda activate dawand
```

### System Requirements
- Python 3.7
- CUDA 11
- 16 GB GPU (for training)
- Blender 3.3.1 (for synthetic dataset generation)

## Training
### Download original training data from paper 
[Synthetic Dataset](https://drive.google.com/file/d/1fWASZLzh85WWLhQi6gYJg1uRJK5m92yK/view?usp=sharing)
[Distortion Self-Supervision Dataset](https://drive.google.com/file/d/1AjfXRL6UhPJSNXlEucDKV0AvsvZx-Uya/view?usp=sharing)

To train, simply provide the path to the dataset folder as inputs to the argument `--dataroot` and the path to the test subfolder to the argument `--test_dir`. See the `scripts` folder for example commands for training the network with the same parameters as in the paper. 
```bash 
# Synthetic Pretraining 
python train_v2.py --gpu_ids 0 --mode evaluation --shuffle_topo --export_preds --export_view_freq 10 --run_test_freq 10 --ncf 16 16 16 16 16 16 --niter_decay 10 --niter 150 --resblocks 3 --lr 0.001 --drop_relu --num_threads 0 --save_latest_freq 10 --save_epoch_freq 2 --selection_module --reweight_loss --dataset_mode intseg --arch intseg --test_dir ./synthetic_dataset/test --export_save_path outputs --dataroot synthetic_dataset --checkpoints_dir outputs --max_sample_size 8000 --supervised --extrinsic_condition_placement pre --batch_size 15 --name pretrain --cachefolder cache --anchorcachefolder anchorcache --resconv --batch_size 8 --leakyrelu --layernorm --dropout --extrinsic_features onehot hks --num_aug 1 --vnormaug --testaug --edgefeatures dihedrals

# Distortion self-supervision 
python train_v2.py --gpu_ids 0 --mode evaluation --shuffle_topo --export_view_freq 4 --run_test_freq 4 --ncf 16 16 16 16 16 16 --niter_decay 10 --niter 150 --resblocks 3 --lr 0.001 --drop_relu --num_threads 0 --save_latest_freq 10 --save_epoch_freq 2 --selection_module --reweight_loss --dataset_mode intseg --arch intseg --resconv --batch_size 8 --leakyrelu --layernorm --load_pretrain --which_epoch best --extrinsic_condition_placement pre --extrinsic_features onehot hks --edgefeatures dihedrals --network_load_path ./outputs/pretrain --segboundary neighbor  --export_preds --distortion_loss count --distortion_metric arap --name dawand --cachefolder cache --anchorcachefolder anchorcache --delayed_distortion_epochs 0 --step2paramloss --export_save_path outputs --checkpoints_dir outputs --supervised --dataroot distortion_dataset --test_dir distortion_dataset/test --mixedtraining --max_grad 5 --num_aug 1 --vnormaug --testaug --cut_param --gcsmoothness --gcsmoothness_weight 0.001 
```

### Generate Synthetic Dataset 
Call the script in `./scripts/generate_synthetic_data.sh` to generate the synthetic dataset using the procedure outlined in the paper. The data will be stored in `./datasets/synthetic_data`. 

To generate the deformed primitives with custom parameters, you will need to have Blender 3.3.1 installed and available in PATH. From there, you can generate random deformed primitives using the primitives provided in `datasets/primitives` by calling 
```bash
blender --background --python blender_deform.py --datadir ./datasets/primitives --outdir ./datasets/deformed_primitives ... 
```
This will generate a set of deformed primitives with the same ground truth near-developable decompositions. Refer to the `blender_deform.py` file for the adjustable parameters for generating the synthetic data. 

After generating the deformed primitives, you can call `generate_synthetic_data.py` with custom parameters to construct the synthetic dataset with randomly sampled selection points and ground truth labels.  

### Build your own natural shape dataset 
The differentiable parameterization layer enables training DA Wand over any arbitrary set of meshes. Creating a dataset for distortion self-supervised training is simple. 

1. Create a folder with the name of the dataset, with subfolders `train` and `test` Split your training meshes between the `train` and `test` folders. 
2. For each mesh in the subfolders, sample selection points as desired into a **Python list** and use `dill` to save them as pickled files in a folder titled `anchors`. The name of the pickled files should be `{name of obj}.pkl`. There should be separate `train/anchors` and `test/anchors` folders. 
3. To incorporate mixed training with ground truth labels, copy the respective labelled meshes into the `train` and `test` folders, copy their anchor `.pkl` files into the `anchors` subfolder, and create a new folder titled `labels` to store the labels. Each label file should be a binary numpy array with the same length as the number of mesh triangles. The label files should be titled by `{name of obj}{anchor index}.npy`, where the anchor index is the index of the respective selection point from the mesh selection point list saved in `anchors`. 
To train with mixed data, simply pass the flags `--supervised --mixedtraining` into the `train.py` call. 

## Interactive Demo 

## Citation
```
@article{liu2022dawand,
         author = {Liu, Richard and Aigerman, Noam and Kim, Vladimir G. and Hanocka, Rana},
         title  = {DA Wand: Distortion-Aware Selection using Neural Mesh Parameterization},
         journal = {arXiv},
         year = {2022}
         }
```
