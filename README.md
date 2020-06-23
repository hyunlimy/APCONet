APCONet: <br/> AI model for arterial pressure waveform derived cardiac output
----------------------------------------------------
This repository is the official python implementation of APCONet-v1.

> Author: [Hyun-Lim Yang](https://sites.google.com/view/hyunlim-yang) 
(*[InfoLab](http://infolab.kaist.ac.kr/)*, DGIST, South Korea) <br/>

> Credits: <br/>
> * Hyung-Chul Lee (*[VitalLab](https://vitaldb.net/)*, SNUH, South Korea)
> * Chul-Woo Jung (*[VitalLab](https://vitaldb.net/)*, SNUH, South Korea)
> * [Min-Soo Kim](http://infolab.kaist.ac.kr/members/Min-Soo%20Kim/) 
(*[InfoLab](http://infolab.kaist.ac.kr/)*, KAIST, South Korea)

APCONet is an end-to-end beat-to-beat deep learning model for Stroke Volume monitoring.<br/>
It predicts Stroke Volume from 100Hz 20-second of arterial blood pressure and demographic information (age, sex, height weights).<br/>

>:heavy_exclamation_mark: **For Vital Recorder User** <br>
> Our program is developed with great thanks to [VitalDB](https://vitaldb.net/) in Seoul National University Hospital, 
> and we provide real-time AWS cloud-base prediction in line with Vital Recorder web monitoring system!<br/>
> Please feel free to try out our works in your clinical practice :) <br/>
> If you are new to Vital Recorder, please refer to the [demo](https://vitaldb.net/web-monitoring/) 
> or figure out what you can do with Vital Recorder from the [Vital Recorder Paper](https://www.nature.com/articles/s41598-018-20062-4). 

## Dataset
We share a de-identified dataset of pre-processed arterial blood pressure waveform and demographic information, 
and its corresponding pulmonary artery catheter derived stroke volume which were used for tuning APCONet.<br/>
Other full data which were used for pre-training is available upon request at [VitalDB databank](https://vitaldb.net/data-bank/).<br/>

## Model Inference
To evaluate the model, you can run the below script. <br/>
`$MODEL_PATH` is the checkpoint of the model used for inference, and `$DATASET_DIR` is the root directory of the preprocessed dataset. <br/>
You can download the pretrained APCONet from [here](). <br/>
Dataset must be numpy array of shape (batch, 2000) and shape of (batch, 4) for wave and demographic information respectively. <br/>
Note that if you do not specify the directory of `$DATASET_DIR`, then the model will automatically used `sample_wave.np` and `sample_ashw.np`.<br/>

```
python3 apconet.py --model_path $MODEl_PATH --dataset_path $DATASET_PATH
```

## Sample Results
![](example_results.png)


## Citation
Not specified yet

