import textgrid
import pandas as pd

def get_onset_offset_phones(textgrid):
    """
    Get onset and offset of each phone in a textgrid
    :param textgrid: TextGrid object
    :return: list of tuples of (phone, onset, offset)
    """

    #get the intervals of phones from the textgrid
    names = [tier.name for tier in textgrid.tiers]
    phone_index = names.index('phones')
    intervals = textgrid.tiers[phone_index].intervals

    #get the onset and offset of each phone
    phones = []
    for interval in intervals:
        minTime, maxTime = interval.minTime, interval.maxTime
        phone = interval.mark
        if phone=='':
            continue
        
        #delete the number at the end of the phone
        if phone[-1].isdigit():
            phone = phone[:-1]

        phones.append((phone,minTime, maxTime))

    return phones
def get_onset_offset_words(textgrid):
    """
    Get onset and offset of each phone in a textgrid
    :param textgrid: TextGrid object
    :return: list of tuples of (phone, onset, offset)
    """

    #get the intervals of phones from the textgrid
    names = [tier.name for tier in textgrid.tiers]
    phone_index = names.index('words')
    intervals = textgrid.tiers[phone_index].intervals

    #get the onset and offset of each phone
    words = []
    for interval in intervals:
        minTime, maxTime = interval.minTime, interval.maxTime
        word = interval.mark
        if word=='':
            continue

        words.append((word,minTime, maxTime))

    return words

def get_onset_offset_phones_words(textgrid):
    """
    Get onset and offset and the word it belongs to of each phone in a textgrid

    :param textgrid: TextGrid object
    :return: list of tuples of (phone, onset, offset, word, word onset, word offset,word index)
    """
    phonemes = get_onset_offset_phones(textgrid)
    words = get_onset_offset_words(textgrid)
    #get the onset and offset of each phone
    for i,phone in enumerate(phonemes):
        minTime, maxTime = phone[1], phone[2]
        for word in words:
            if word[1]<=minTime and word[2]>=maxTime:
                phonemes[i] = phonemes[i]+(word[0],word[1],word[2],words.index(word))

                # compute relative onset and offset of the phone
                phonemes[i] = phonemes[i]+(minTime-word[1],maxTime-word[1])
                break
        assert len(phonemes[i])==9, 'phoneme can not be aligned with word'
    return phonemes

def parse_data(csv_path, label_col = 0, onset_col = 1, offset_col = None):
    """
    Parse data from a CSV file and return a list of intervals with their corresponding labels.

    Args:
    - csv_path (str): The path to the CSV file.
    - label_col (int): The index of the column containing the labels. Default is 0.
    - onset_col (int): The index of the column containing the onset times. Default is 1.
    - offset_col (int): The index of the column containing the offset times. Default is None.

    Returns:
    - intervals (DataFrame): A DataFrame containing the intervals and labels.
    """
    csv = pd.read_csv(csv_path)

    intervals = []
    
    for i in range(len(csv) - 1):
        parts = csv.iloc[i]
        label = "" if parts[label_col] == ">" else parts[label_col]
        start = float(parts[onset_col])
        if offset_col is None:
            end = float(csv.iloc[i+1][1])
        else:
            end = float(parts[offset_col])
        intervals.append({"onset": start, "offset": end, "label": label})
    if offset_col is not None:
        start = float(csv.iloc[-1][onset_col])
        end = float(csv.iloc[-1][offset_col])
        label = "" if csv.iloc[-1][label_col] == ">" else csv.iloc[-1][label_col]
        intervals.append({"onset": start, "offset": end, "label": label})
    
    intervals = pd.DataFrame(intervals)
    
    return intervals

def create_textgrid_tier(name, intervals):
    """
    Creates a TextGrid tier with the given name and intervals.
    Returns a string representation of the tier.
    """

    tier = [
        f'        class = "IntervalTier"', 
        f'        name = "{name}"', 
        '        xmin = 0', 
        f'        xmax = {intervals.iloc[-1]["offset"]}', 
        f'        intervals: size = {len(intervals)}'
    ]
    
    for i in range(len(intervals)):
        interval = intervals.iloc[i]
        tier.append(f'        intervals [{i+1}]:')
        tier.append(f'            xmin = {interval["onset"]}')
        tier.append(f'            xmax = {interval["offset"]}')
        tier.append(f'            text = "{interval["label"]}"')
    
    return "\n".join(tier)

def intervaldf_to_textgrid(interval_dfs, out_path , names=['phones','words']):
    tiers = []
    maximum = 0
    for interval_df,name in zip(interval_dfs,names):
        tier = create_textgrid_tier(name, interval_df)
        tiers.append(tier)
        maximum = max(maximum,interval_df.iloc[-1]["offset"])
    n_tiers = len(tiers)
    textgrid_content0 = [
    'File type = "ooTextFile"',
    'Object class = "TextGrid"',
    '',
    'xmin = 0',
    f'xmax = {maximum}',
    'tiers? <exists>',
    f'size = {n_tiers}',
    'item []:']
    textgrid_content1 = [ f'    item [{i+1}]:\n{tier}' for i,tier in enumerate(tiers) ]
    textgrid_content2 = ['']
    
    textgrid_content = textgrid_content0+textgrid_content1+textgrid_content2

    textgrid_str = "\n".join(textgrid_content)

    # Saving the TextGrid content to a file
    textgrid_file_path = out_path
    with open(textgrid_file_path, 'w') as file:
        file.write(textgrid_str)


def csv_to_textgrid(csv_paths, out_path , names=['phones','words'], label_col = 0, onset_col = 1, offset_col = None):
    """
    Converts two CSV files containing interval data (one for phonemes and one for words)
    into a TextGrid file and saves it to the specified output path.

    Args:
        phoneme_csv_path (str): Path to the CSV file containing phoneme interval data.
        word_csv_path (str): Path to the CSV file containing word interval data.
        out_path (str): Path to save the resulting TextGrid file.

    Returns:
        None
    """
    interval_dfs = [parse_data(csv_path, label_col, onset_col, offset_col) for csv_path in csv_paths]
    intervaldf_to_textgrid(interval_dfs, out_path, names)
    


# if __name__ == '__main__':
#     import glob
#     import os
#     import csv
#     import scipy.io as sio
#     import pickle
#     textgrids = glob.glob('/scratch/snormanh_lab/shared/Sigurd/ljs/projects/dana/dana_aligned/*.TextGrid')
#     for textgrid_file in textgrids:
#         tg = textgrid.TextGrid.fromFile(textgrid_file)
        # phones = [('phoneme','onset','offset')]+get_onset_offset_phones(tg)
        # #save as a pkl file
        # pkl_file = textgrid_file.replace('.TextGrid','_phoneme.pkl')
        # with open(pkl_file,'wb') as f:
        #     pickle.dump(phones,f)
        #save as a matlab file
        # mat_file = textgrid_file.replace('.TextGrid','_phoneme.mat')
        # sio.savemat(mat_file,{'phones':phones})
        # words = [('word','onset','offset')]+get_onset_offset_words(tg)
        # #save as a pkl file
        # pkl_file = textgrid_file.replace('.TextGrid','_word.pkl')
        # with open(pkl_file,'wb') as f:
        #     pickle.dump(words,f)
        #save as a matlab file
        # mat_file = textgrid_file.replace('.TextGrid','_word.mat')
        # sio.savemat(mat_file,{'words':words})
        
        # phones_words = [('phoneme','phoneme_absolute_onset','phoneme_absolute_offset','word','word_absolute_onset','word_absolute_offset','word_index','phoneme_relative_onset','phoneme_relative_offset')]+get_onset_offset_phones_words(tg)
        # #save as a pkl file
        # pkl_file = textgrid_file.replace('.TextGrid','_phoneme_word.pkl')
        # with open(pkl_file,'wb') as f:
        #     pickle.dump(phones_words,f)
        # #save as a matlab file
        # mat_file = textgrid_file.replace('.TextGrid','_phoneme_word.mat')
        # sio.savemat(mat_file,{'phones_words':phones_words})
        # #save as a csv file
        # csv_file = textgrid_file.replace('.TextGrid','_phoneme_word.csv')
        # with open(csv_file,'w') as f:
        #     writer = csv.writer(f)
        #     writer.writerows(phones_words)



    # for textgrid_file in textgrids:
    #     tg = textgrid.TextGrid.fromFile(textgrid_file)
    #     words = [('word','onset','offset')]+get_onset_offset_words(tg)

        #save as a matlab file
        # mat_file = textgrid_file.replace('.TextGrid','_word.mat')
        # sio.savemat(mat_file,{'words':words})
        





if __name__ == '__main__':

    csv_paths = ['/scratch/snormanh_lab/shared/projects/music-iEEG/origin_notes/slakh_60sec_excerpts_variation_v4/test/Track01876/all_src/test_Track01876_async_variation.csv']
    out_path = '/home/gliao2/my_scratch/dev/test_Track01876_async_variation.textgrid'

    label_col=6
    onset_col=0
    offset_col=1
    interval_df = parse_data(csv_paths[0], label_col, onset_col, offset_col)
    interval_dfs = []
    for label in interval_df['label'].unique():
        interval_dfs.append(interval_df[interval_df['label']==label])
    names = interval_df['label'].unique()
    intervaldf_to_textgrid(interval_dfs, out_path, names)