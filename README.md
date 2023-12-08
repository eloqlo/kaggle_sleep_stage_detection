# kaggle_sleep_stage_detection
- https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/overview
- 2023.12.07

## My Trials

1. Start baseline with naive ML approach (Score 0.279)
2. Set the task as classification for solving this problem (Failed)
3. Set the task as segmentation and adopt proper paper's model : [U-Time](https://arxiv.org/abs/1910.11162) (Score 0.449)
4. Set the task as regression and try various approaches (Score 0.664)
   - data (preprocessing, feature engineering)
   - model (Feature Extractor, Encoder, Decoder)

## My best Model Combination was

**Feature Extractor**(1D convolutions)  
**Encoder** (Downsampling, Self Attention, Upsampling)  
**Decoder** (1D Convolutions)  

* I tried to focus on processing different pattern of input data feature. So I used Channel-Wise Covolution also, but didn't have much time to do a proper experiments.
