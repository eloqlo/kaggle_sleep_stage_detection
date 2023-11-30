import pandas as pd
import numpy as np

def predict(id_:str, pred:np.array, th=0.5, itv_hour=12) -> pd.DataFrame:
    
    # pred (seq, 2)
    result=pd.DataFrame(columns=['series_id','step','event','score'])
    itv_step=itv_hour*60*12
    onset_max_score=0
    onset_step=0
    wakeup_max_score=0
    wakeup_step=0
    for step in range(len(pred)):
        
        # scan max point.
        if pred[step][0]>th and pred[step][0]>onset_max_score:
            onset_step=step
            onset_max_score=pred[step][0]
        if pred[step][1]>th and pred[step][1]>wakeup_max_score:
            wakeup_step=step
            wakeup_max_score=pred[step][1]
            
        # if there are no new point between interval, save that point as a prediction output.
        if step-onset_step>itv_step and onset_max_score!=0:
            result.loc[len(result)] = [id_, onset_step, 'onset', onset_max_score]
            onset_step=0
            onset_max_score=0
        if step-wakeup_step>itv_step and wakeup_max_score!=0:
            result.loc[len(result)] = [id_, wakeup_step, 'wakeup', wakeup_max_score]
            wakeup_step=0
            wakeup_max_score=0
        
        # just in case, if max point is in the end of the prediction.
        if step==len(pred)-1 and onset_max_score!=0:
            result.loc[len(result)] = [id_, onset_step, 'onset', onset_max_score]
        if step==len(pred)-1 and wakeup_max_score!=0:
            result.loc[len(result)] = [id_, wakeup_step, 'wakeup', wakeup_max_score]
    
    return result
