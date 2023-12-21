# kaggle_sleep_stage_detection
- https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/overview
- 2023.12.07

## 🤖 My Trials were...

1. Start baseline with XGBoost (**Score 0.279**)
2. Set the task as "classification" for solving this problem (**Failed**)
3. Set the task as "segmentation" and adopt proper paper's model : [U-Time](https://arxiv.org/abs/1910.11162) (**Score 0.449**)
4. Set the task as "regression" and try various approaches(focal loss, gaussian target) (**Score 0.664** best!)
   - data (preprocessing, feature engineering)
   - model (Feature Extractor, Encoder, Decoder)

## 🤖 My best Model Combination was...

**Feature Extractor**(1D CNN)  
**Encoder** (Downsampling, Attention, Upsampling)  
**Decoder** (1D CNN)  

* I tried to focus on processing different pattern of input data feature. So I used Channel-Wise Covolution also, but didn't have much time to do a proper experiments.
