import textgrid
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

if __name__ == '__main__':
    import glob
    import os
    import csv
    import scipy.io as sio
    import pickle
    textgrids = glob.glob('/scratch/snormanh_lab/shared/Sigurd/ljs/dana_aligned/*.TextGrid')
    for textgrid_file in textgrids:
        tg = textgrid.TextGrid.fromFile(textgrid_file)
        phones = [('phoneme','onset','offset')]+get_onset_offset_phones(tg)
        # #save as a pkl file
        # pkl_file = textgrid_file.replace('.TextGrid','_phoneme.pkl')
        # with open(pkl_file,'wb') as f:
        #     pickle.dump(phones,f)
        #save as a matlab file
        mat_file = textgrid_file.replace('.TextGrid','_phoneme.mat')
        sio.savemat(mat_file,{'phones':phones})
        words = [('word','onset','offset')]+get_onset_offset_words(tg)
        # #save as a pkl file
        # pkl_file = textgrid_file.replace('.TextGrid','_word.pkl')
        # with open(pkl_file,'wb') as f:
        #     pickle.dump(words,f)
        #save as a matlab file
        mat_file = textgrid_file.replace('.TextGrid','_word.mat')
        sio.savemat(mat_file,{'words':words})
        
    # for textgrid_file in textgrids:
    #     tg = textgrid.TextGrid.fromFile(textgrid_file)
    #     words = [('word','onset','offset')]+get_onset_offset_words(tg)

        #save as a matlab file
        # mat_file = textgrid_file.replace('.TextGrid','_word.mat')
        # sio.savemat(mat_file,{'words':words})
