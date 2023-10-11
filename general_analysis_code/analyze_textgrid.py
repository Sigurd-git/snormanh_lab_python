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




if __name__ == '__main__':
    import glob
    import os
    import csv
    import scipy.io as sio
    import pickle
    textgrids = glob.glob('/scratch/snormanh_lab/shared/Sigurd/ljs/projects/dana/dana_aligned/*.TextGrid')
    for textgrid_file in textgrids:
        tg = textgrid.TextGrid.fromFile(textgrid_file)
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
        
        phones_words = [('phoneme','phoneme_absolute_onset','phoneme_absolute_offset','word','word_absolute_onset','word_absolute_offset','word_index','phoneme_relative_onset','phoneme_relative_offset')]+get_onset_offset_phones_words(tg)
        #save as a pkl file
        pkl_file = textgrid_file.replace('.TextGrid','_phoneme_word.pkl')
        with open(pkl_file,'wb') as f:
            pickle.dump(phones_words,f)
        #save as a matlab file
        mat_file = textgrid_file.replace('.TextGrid','_phoneme_word.mat')
        sio.savemat(mat_file,{'phones_words':phones_words})
        #save as a csv file
        csv_file = textgrid_file.replace('.TextGrid','_phoneme_word.csv')
        with open(csv_file,'w') as f:
            writer = csv.writer(f)
            writer.writerows(phones_words)



    # for textgrid_file in textgrids:
    #     tg = textgrid.TextGrid.fromFile(textgrid_file)
    #     words = [('word','onset','offset')]+get_onset_offset_words(tg)

        #save as a matlab file
        # mat_file = textgrid_file.replace('.TextGrid','_word.mat')
        # sio.savemat(mat_file,{'words':words})
        


def parse_data(csv_path):
    """
    Parses a CSV file containing interval data and returns a list of dictionaries,
    where each dictionary represents an interval and has keys "xmin" and "xmax".
    """
    csv = pd.read_csv(csv_path)

    intervals = []
    
    for i in range(len(csv) - 1):
        parts = csv.iloc[i]
        text = "" if parts[0] == ">" else parts[0]
        start = float(parts[1])
        end = float(csv.iloc[i+1][1])
        intervals.append({"xmin": start, "xmax": end, "text": text})
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
        f'        xmax = {intervals[-1]["xmax"]}', 
        f'        intervals: size = {len(intervals)}'
    ]
    
    for i, interval in enumerate(intervals):
        tier.append(f'        intervals [{i+1}]:')
        tier.append(f'            xmin = {interval["xmin"]}')
        tier.append(f'            xmax = {interval["xmax"]}')
        tier.append(f'            text = "{interval["text"]}"')
    
    return "\n".join(tier)




def csv_to_textgrid(phoneme_csv_path, word_csv_path, out_path):
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
    phoneme_intervals = parse_data(phoneme_csv_path)
    word_intervals = parse_data(word_csv_path)
    phoneme_tier = create_textgrid_tier("phones", phoneme_intervals)
    word_tier = create_textgrid_tier("words", word_intervals)

    textgrid_content = [
    'File type = "ooTextFile"',
    'Object class = "TextGrid"',
    '',
    'xmin = 0',
    f'xmax = {phoneme_intervals[-1]["xmax"]}',
    'tiers? <exists>',
    'size = 2',
    'item []:',
    f'    item [1]:\n{word_tier}',
    f'    item [2]:\n{phoneme_tier}',
]

    textgrid_str = "\n".join(textgrid_content)

    # Saving the TextGrid content to a file
    textgrid_file_path = out_path
    with open(textgrid_file_path, 'w') as file:
        file.write(textgrid_str)


if __name__ == '__main__':

    phoneme_csv_path = '/Users/sigurd/Desktop/natural-fast.csv'
    word_csv_path = '/Users/sigurd/Desktop/natural-fast-word.csv'
    out_path = 'test1.TextGrid'
    
    csv_to_textgrid(phoneme_csv_path, word_csv_path, out_path)