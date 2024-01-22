# Ship-Go: SAR Ship Images Inpainting via Instance-to-Image Generative Diffusion Models

[Paper](https://www.sciencedirect.com/science/article/pii/S0924271623003350?via%3Dihub) | 
## Introduce

This is an official implementation of **Ship-Go: SAR Ship Images Painting via Instance-to-Image Generative Diffusion Models** by **Pytorch**, the code template is from the project: [Plattle](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models).

### Pre-trained Model

| Dataset   | Task       |  Epochs    | GPUs×Days×Bs | URL                                                          |
| --------- | ---------- | ---------- | ------------ | ------------------------------------------------------------ |
| SSDD      | Painting   | 5000       | 1×5×2        | [Google Drive](https://drive.google.com/drive/folders/1ZhGBmnmGNdDClcEhsAUMR3SdxK3IM-PQ) |

**Bs** indicates sample size per gpu.

### Data Prepare

- [SSDD](https://github.com/TianwenZhang0825/Official-SSDD)
- [HRSID](https://github.com/chaozhong2010/HRSID) 

### Config file selection
For SSDD dataset, please select the config file "config/sard.json"

For HRSID dataset, please select the config file "config/sard_hrsid.json"

### Training/Resume Training
1. Download the checkpoints from given links.
2. Set `resume_state` of configure file to the directory of previous checkpoint. Take the following as an example, this directory contains training states and saved model:

```yaml
"path": { //set every part file path
	"resume_state": "experiments/ssdd/checkpoint/5000" 
},
```
2. Run the script:

```python
python run.py -p train -c config/sard.json
```

### Test

1. Modify the configure file to point to your data following the steps in **Data Prepare** part.
2. Set your model path following the steps in **Resume Training** part.
3. Run the script:
```python
python run.py -p test -c config/sard.json
```


## Citation

```
@article{zhang2024ship,
  title={Ship-Go: SAR Ship Images Inpainting via instance-to-image Generative Diffusion Models},
  author={Zhang, Xin and Li, Yang and Li, Feng and Jiang, Hangzhi and Wang, Yanhua and Zhang, Liang and Zheng, Le and Ding, Zegang},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={207},
  pages={203--217},
  year={2024},
  publisher={Elsevier}
}
```

