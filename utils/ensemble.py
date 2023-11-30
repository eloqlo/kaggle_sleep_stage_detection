import pandas as pd
import numpy as np

"""
input:  models, options, dataset
output:  [(id,pred),(id,pred),...]  << 앙상블 결과

models:     [[(model1,opt),(model2,opt), ...], [(model#,opt), ...]]
    - models.opt:      (use_mse<bool>, use_hour_emb<bool>, ensemble_w<int>)
ds_opts:        [[ds_type, use_feat, use_hour, make_hour_emb]*N]
ds:        <nn.utils.data.Dataset>
"""
def ensemble_models(models:list,ds_opts:list, ds:Dataset) -> list:
    preds=[]
    result=[]
    # ds 순회
    for i, opt in enumerate(ds_opts):
        preds_ds=[] # (id,pred)
        preds_ds_id=[]
        ds.change_mode(*opt)
        dl = DataLoader(ds, batch_size=1, shuffle=False)

        prev_id=None
        for k,(x,h,y,(id_,id_idx)) in enumerate(dl):
            cur_id = id_[0]
            pred_ds_model=[]
            ensemble_scaler=0
            for model,model_opt in models[i]:
                use_mse=model_opt[0]
                use_hour_emb=model_opt[1]
                ensemble_w = model_opt[2]
                x=x.to(torch.float32).to(device)
                if use_hour_emb:
                    h=h.to(device)
                with torch.no_grad():
                    model.eval()
                    if use_hour_emb:
                        pred = model(x,h).to('cpu').numpy()
                    else:
                        pred = model(x).to('cpu').numpy()
                    if use_mse:
                        # sigmoid
                        pred = 1/(1 + np.exp(-pred))
                # ensemble
                pred_ds_model.append(pred*ensemble_w)
                ensemble_scaler+=ensemble_w
            pred_seg = sum(pred_ds_model)/ensemble_scaler # ensemble for one sample's segment.
            pred_ds_model=[]
            
            # id에 대해 sample 하나 완성했다면, save.
            if k!=0 and prev_id!=cur_id:
                pred_complete = np.concatenate(preds_ds_id).reshape(-1,2)
                preds_ds.append({'id':prev_id, 'ensembled_pred':pred_complete})
                preds_ds_id=[]
                
            preds_ds_id.append(pred_seg)
            
            if k==len(dl)-1:
                pred_complete = np.concatenate(preds_ds_id).reshape(-1,2)
                preds_ds.append({'id':cur_id, 'ensembled_pred':pred_complete})
                preds_ds_id=[]
            
            prev_id=cur_id
        preds.append(preds_ds)
    
    # last ensemble
    for ensemble_middle_values in zip(*preds):
        ids=[]
        ensembled_pred=[]
        for ele in ensemble_middle_values:
            ids.append(ele['id'])
            ensembled_pred.append(ele['ensembled_pred'])
        if len(set(ids))!=1:
            raise ValueError("all ids should be same. debug it!")
        else:
            final_ensemble_pred = sum(ensembled_pred)/len(ensemble_middle_values)
            result.append((ids[0],final_ensemble_pred))
            
    return result # [(id,pred_ensembled)*N_ids]
