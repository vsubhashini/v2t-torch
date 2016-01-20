# convert raw data to hdf5 dataset

import os
import json
import argparse
import h5py
import glob
import numpy as np
from scipy.misc import imread, imresize

vid_dir = ''
label_dir = ''
save_dir = ''

def build_vocab(path_to_vocab):

    ix_to_word = {}
    word_to_ix = {}

    vocab_count = 0
    with open(path_to_vocab, 'r') as f:
        for word in f:
            word = word.strip()
            vocab_count = vocab_count+1
            ix_to_word[vocab_count] = word
            word_to_ix[word] = vocab_count

    return ix_to_word, word_to_ix, vocab_count

def assign_split(split_name):
    if split_name == 'train':
        return 1, 1200, 'labels_train.txt'
    else:
        return 1201, 1300, 'labels_val.txt'

def get_max_frames(vid_start, vid_end):
    max_frames = 0
    for i in xrange(1, vid_end-vid_start+1+1):
        num_frames = len(glob.glob(os.path.join(vid_dir, 'vid' + str(i+vid_start-1), '*')))
        if num_frames > max_frames:
            max_frames = num_frames
    return max_frames

def get_sent_info(sent_file):
    sent_count = 0
    longest_sent = 0
    with open(os.path.join(label_dir, sent_file), 'r') as f:
        for lines in f:
            sent_count = sent_count + 1
            caption = lines.split('\t')[1]
            caption_length = len(caption.split(' '))

            if caption_length > longest_sent:
                longest_sent = caption_length

    return sent_count, longest_sent

def main(params):
    
    vid_start, vid_end, sent_file = assign_split(params['split_name'])

    print('loading vocabulary...')
    ix_to_word, word_to_ix, vocab_size = build_vocab(params['vocab_file'])

    print('processing sentences...')
    num_sent, longest_sent = get_sent_info(sent_file)
    
    sentences = np.zeros((num_sent, longest_sent+1), dtype='uint32')
    label_length = np.zeros((num_sent), dtype='uint32')
    vid_ids = np.zeros((num_sent), dtype='uint32')
    start_ix = np.zeros((vid_end-vid_start+1), dtype='uint32')
    end_ix = np.zeros((vid_end-vid_start+1), dtype='uint32')

    vid_id = ''
    vid_count = 0
    sent_count = 0
    with open(os.path.join(label_dir, sent_file), 'r') as f:
        for lines in f:
            sent_count = sent_count+1
            s = lines.strip().split('\t')
            if s[0] != vid_id:
                if sent_count != 1:
                    end_ix[vid_count - 1] = sent_count - 1
                start_ix[vid_count] = sent_count

                vid_count = vid_count + 1
                vid_id = s[0]

            words = s[1].split()
            for ix, word in enumerate(words):
                sentences[sent_count-1, ix] = word_to_ix[word]
            sentences[sent_count-1, len(words)] = vocab_size+1

            label_length[sent_count-1] = len(words)+1
            vid_ids[sent_count-1] = vid_count
    end_ix[-1] = sent_count


    print('processing videos...')
    if params['split_name'] == 'train':
        max_frames = params['max_frames']
    else:
        max_frames = get_max_frames(vid_start, vid_end)

    videos = np.zeros((vid_end-vid_start+1, max_frames, 3, 256, 256), dtype='uint8')
    video_length = np.zeros((vid_end-vid_start+1), dtype='uint32')

    for i in xrange(1, vid_end-vid_start+1+1):
        print('{:d}/{:d}'.format(i, vid_end-vid_start+1))
        num_frames = len(glob.glob(os.path.join(vid_dir, 'vid' + str(i+vid_start-1), '*')))
        if num_frames > max_frames:
            num_frames = max_frames
        video_length[i-1] = num_frames

        for frame_num in xrange(1, num_frames):
            I = imread(os.path.join(vid_dir, 'vid' + str(i+vid_start-1),'vid{:d}_frame{:03d}.jpg'.format(i+vid_start-1, frame_num)))
            Ir = imresize(I, (256, 256))
            Ir = Ir.transpose(2,0,1)
            videos[i-1, frame_num-1] = Ir

    f = h5py.File(os.path.join(save_dir, '{}.h5'.format(params['split_name'])), 'w')
    f.create_dataset("videos", dtype='uint8', data=videos)
    f.create_dataset("video_length", dtype='uint32', data=video_length)
    f.create_dataset("labels", dtype='uint32', data=sentences)
    f.create_dataset("label_length", dtype='uint32', data=label_length)
    f.create_dataset("label_start_ix", dtype='uint32', data=start_ix)
    f.create_dataset("label_end_ix", dtype='uint32', data=end_ix)
    # f.create_dataset("label_to_id", dtype='uint32', data=vid_ids)
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--split_name', default='train', type=str, help='train|val')
    parser.add_argument('--max_frames', default=60, type=int, help='max number of frames for each instance (-1 for no limit)')
    parser.add_argument('--video_dir', default='', type=str, help='where to load videos from')
    parser.add_argument('--label_dir', default='', type=str, help='where to load labels from')
    parser.add_argument('--vocab_file', default='', type=str, help='path to vocab_file')
    parser.add_argument('--save_dir', default='', type=str, help='where to save preprocessed data to')

    args = parser.parse_args()
    params = vars(args)
    
    vid_dir = params['video_dir']
    label_dir = params['label_dir']
    save_dir = params['save_dir']

    main(params)