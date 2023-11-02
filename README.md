# Ship-Go: SAR Ship Images Painting via Instance-to-Image Generative Diffusion Models

[Paper](... ) | 
## Introduce

This is an official implementation of **Ship-Go: SAR Ship Images Painting via Instance-to-Image Generative Diffusion Models** by **Pytorch**, the code template is from the project: [Plattle](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models).

## Status

## Results

The DDPM model requires significant computational resources, and we have only built a few example models to validate the ideas in this paper.

### Visuals


### Pre-trained Model

| Dataset   | Task       |  Epochs    | GPUs×Days×Bs | URL                                                          |
| --------- | ---------- | ---------- | ------------ | ------------------------------------------------------------ |
| SSDD      | Painting   | 5000       | 1×5×2        | [Google Drive](https://drive.google.com/drive/folders/1ZhGBmnmGNdDClcEhsAUMR3SdxK3IM-PQ) |

**Bs** indicates sample size per gpu.



### Data Prepare

- [SSDD](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256)
- [HRSID](http://places2.csail.mit.edu/download.html) | [Places2 Kaggle](https://www.kaggle.com/datasets/nickj26/places2-mit-dataset?resource=download)



### Training/Resume Training
1. Download the checkpoints from given links.
1. Set `resume_state` of configure file to the directory of previous checkpoint. Take the following as an example, this directory contains training states and saved model:

```yaml
"path": { //set every part file path
	"resume_state": "experiments/inpainting_celebahq_220426_150122/checkpoint/100" 
},
```
2. Set your network label in `load_everything` function of `model.py`, default is **Network**. Follow the tutorial settings, the optimizers and models will be loaded from 100.state and 100_Network.pth respectively.

```python
netG_label = self.netG.__class__.__name__
self.load_network(network=self.netG, network_label=netG_label, strict=False)
```

3. Run the script:

```python
python run.py -p train -c config/inpainting_celebahq.json
```

We test the U-Net backbone used in `SR3` and `Guided Diffusion`,  and `Guided Diffusion` one have a more robust performance in our current experiments.  More choices about **backbone**, **loss** and **metric** can be found in `which_networks`  part of configure file.

### Test

1. Modify the configure file to point to your data following the steps in **Data Prepare** part.
2. Set your model path following the steps in **Resume Training** part.
3. Run the script:
```python
python run.py -p test -c config/inpainting_celebahq.json
```


# Ship-Go
# Ship-Go
