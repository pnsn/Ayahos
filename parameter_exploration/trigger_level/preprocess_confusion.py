import os, pandas, glob, time
import numpy as np

def assign_confusion_matrix_line(row, pick_tol_samp=100):
    """
    Based on extracted features from the trigger_scan.py script, assign
    values for the confusion matrix

        | TP | FN |
        | FP | TN |
    TP = true positive, value 1 assigned if
        A) an analyst pick is contained within a trigger
        B) an analyst pick falls within pick_tol_samp samples of a trigger's peak
         NOTE: analyst picks on one seismic sensor are considered shared with co-located sensors
            e.g., a pick on UW.GNW.--.BH? is also considered a pick on UW.GNW.--.EN?
    TN = true negative, value 1 assigned if:
        A) on data without an associated analyst pick, no triggers occur
        B) on data with an analyst pick, only one trigger occurs and conforms to the requirements of TP above
    FP = false positive, value assigned as:
        A) FP = T triggers for data with an associated pick where all triggers fail to meet
            requirements of TP above
        B) FP = T - 1 as in the case above, but one trigger conforms to TP requirements
        C) FP = T triggers for data without an associated pick
    FN = false negative, value 1 assigned if data have an associated pick but no triggers satisfy
        the requirements of TP above.
    
    :: INPUTS ::
    :param row: [pandas.Series] a series representation of a row from the output CSV files
                produced by trigger_scan.py
    :param pick_tol_samp: [int] pick tolerance samples to allowed for TP, definition B above.

    :: OUTPUTS ::
    :return TP: [int] number of true positive values
    :return FP: [int] number of false positive values
    :return TN: [int] number of true negative values
    :return FN: [int] number of false negative values
    :return tol_needed: [bool] - flag if the tolerance bound inclusion clause was used for TP
    :return labeled_data: [bool] - flag if the C matrix values were determined for labeled or unlabeled data
    """
    # If labeled data
    if np.isfinite(row.arid):
        labeled_data=True
        # If no triggers
        if row.trigger_ct == 0:
            TP = 0
            FN = 1
            FP = 0
            TN = 0
            tol_needed = None
        # If there are triggers
        if row.trigger_ct > 0:
            # Trigger includes pick time - TODO: how to handle long triggers???
            if row.pick_in_trigger:
                TP = 1
                FN = 0
                tol_needed = False
            # Trigger meets pick when including tolerance
            elif abs(row.nearest_peak_dsmp) <= pick_tol_samp:
                TP = 1
                FN = 0
                tol_needed = True
            # Triggers exist, but do not meet pick/tolerance
            else:
                TP = 0
                FN = 1
                tol_needed = None
            # Assign number of false positives
            FP = row.trigger_ct - TP
            # If no false positives, grant this one true negative score
            if FP == 0:
                TN = 1
            else:
                TN = 0
    # If no labeled data are present
    else:
        labeled_data = False
        tol_needed = None
        if row.trigger_ct == 0:
            TN = 1
            FP = 0
            TP = 0
            FN = 0
        else:
            FP = row.trigger_ct
            TN = 0
            TP = 0
            FN = 0
    return TP, FP, TN, FN, tol_needed, labeled_data



pick_tol_samp = 100
PROOT = os.path.join('/Users','nates','Documents','Conferences','2024_SSA','PNSN')
PRED = os.path.join(PROOT,'processed_data','avg_pred_stacks')
INPUT = os.path.join(PRED,'initial_trigger_assessment_sum_v2_Pthresh_*.csv')
inputs = glob.glob(INPUT)
inputs.sort()            

tick = time.time()
for _i, input in enumerate(inputs):
    print(f'processing {input} | ({_i+1}/{len(inputs)}) | et: {(time.time() - tick)/60:.2f} min')
    # Load trigger sweep data for given threshold
    df = pandas.read_csv(input)
    # Grab threshold value
    Pthresh = df.thresh.unique()[0]
    # Process assessment for each line
    holder = []
    for index, row in df.iterrows():
        holder.append(list(assign_confusion_matrix_line(row, pick_tol_samp=pick_tol_samp)))
    df_cm = pandas.DataFrame(holder, columns=['TP','FP','TN','FN','tol_needed','labeled_data'], index=df.index)
    df = pandas.concat([df, df_cm], ignore_index=False, axis=1)
    name, ext = os.path.splitext(input)
    outname = f'{name}_scored.csv'
    df.to_csv(outname, header=True, index=False)


