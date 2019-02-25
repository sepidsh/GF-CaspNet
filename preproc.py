from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
from datetime import datetime
import os
import random
import sys
import threading
import numpy as np
import tensorflow as tf
import json
import matplotlib as mp

import matplotlib.pyplot as plt


sys.path.append(os.curdir)
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)
import cv2
import detect_face as FD2
import math


RESIZE_HEIGHT = 256
RESIZE_WIDTH = 256

tf.app.flags.DEFINE_string('fold_dir', '/home/dpressel/dev/work/AgeGenderDeepLearning/Folds/train_val_txt_files_per_fold/test_fold_is_0',
                           'Fold directory')

tf.app.flags.DEFINE_string('data_dir', '/data/xdata/age-gender/aligned',
                           'Data directory')


tf.app.flags.DEFINE_string('output_dir', '/home/dpressel/dev/work/AgeGenderDeepLearning/Folds/tf/test_fold_is_0',
                           'Output directory')


tf.app.flags.DEFINE_string('train_list', 'age_train.txt',
                           'Training list')
tf.app.flags.DEFINE_string('valid_list', 'age_val.txt',
                           'Test list')

tf.app.flags.DEFINE_integer('train_shards', 10,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('valid_shards', 2,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 2,
                            'Number of threads to preprocess the images.')


FLAGS = tf.app.flags.FLAGS
FACE_PAD = 50

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        
def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(filename, image_buffer, label, height, width):
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/class/label': _int64_feature(label),
        'image/filename': _bytes_feature(str.encode(os.path.basename(filename))),
        'image/encoded': _bytes_feature(image_buffer),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width)
    }))
    return example
    
class ImageCoder(object):
    
    def __init__(self):
        self._sess = tf.Session()
        
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)
        
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
        cropped = tf.image.resize_images(self._decode_jpeg, [RESIZE_HEIGHT, RESIZE_WIDTH])
        cropped = tf.cast(cropped, tf.uint8) 
        self._recoded = tf.image.encode_jpeg(cropped, format='rgb', quality=100)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})
        
    def resample_jpeg(self, image_data):
        image = self._sess.run(self._recoded, #self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})

        return image
        

def _is_png(filename):
    return '.png' in filename



def _process_image(filename, coder):
   
    #img=cv2.imread(filename)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #detector=dlib.get_frontal_face_detector()
    #faces = detector(gray, 1)
    
    #roi_color=img  
    #for (i, rect) in enumerate(faces):
    #    x = rect.left()
    #    y = rect.top()
    #    w = rect.right() - x
    #    h = rect.bottom() - y
    #    upper_cut = [min(img.shape[0], y + h + FACE_PAD), min(img.shape[1], x + w + FACE_PAD)]
    #    lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
    #    roi_color = img[lower_cut[0]:upper_cut[0], lower_cut[1]:upper_cut[1]]
        
    outfile=filename
   #print(outfile)
    #cv2.imwrite(outfile,roi_color)

    filename=outfile
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    image = coder.resample_jpeg(image_data)
    return image, RESIZE_HEIGHT, RESIZE_WIDTH

def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               labels, num_shards):
    
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
    
    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        
        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = int(labels[i])

            image_buffer, height, width = _process_image(filename, coder)
            
            example = _convert_to_example(filename, image_buffer, label,
                                          height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()

def _process_image_files(name, filenames, labels, num_shards):
   
    assert len(filenames) == len(labels)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    coord = tf.train.Coordinator()
    
    coder = ImageCoder()

    threads = []
    for thread_index in xrange(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames, labels, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()

def _find_image_files(list_file, data_dir):
    print('Determining list of input files and labels from %s.' % list_file)
    files_labels = [l.strip().split(' ') for l in tf.gfile.FastGFile(
        list_file, 'r').readlines()]

    labels = []
    filenames = []

    label_index = 1
    
    for path, label in files_labels:
        jpeg_file_path = '%s/%s' % (data_dir, path)
        if os.path.exists(jpeg_file_path):
            filenames.append(jpeg_file_path)
            labels.append(label)

    unique_labels = set(labels)
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)
    
    filenames = [filenames[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]
    
    print('Found %d JPEG files across %d labels inside %s.' %
          (len(filenames), len(unique_labels), data_dir))
    return filenames, labels


def _process_dataset(name, filename, directory, num_shards):
    
    filenames, labels = _find_image_files(filename, directory)
    _process_image_files(name, filenames, labels, num_shards)
    unique_labels = set(labels)
    return len(labels), unique_labels

def main(unused_argv):
    assert not FLAGS.train_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not FLAGS.valid_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with '
        'FLAGS.valid_shards')
    print('Saving results to %s' % FLAGS.output_dir)

    if os.path.exists(FLAGS.output_dir) is False:
        print('creating %s' % FLAGS.output_dir)
        os.makedirs(FLAGS.output_dir)

    # Run it!
    valid, valid_outcomes = _process_dataset('validation', '%s/%s' % (FLAGS.fold_dir, FLAGS.valid_list), FLAGS.data_dir,
                     FLAGS.valid_shards)
    train, train_outcomes = _process_dataset('train', '%s/%s' % (FLAGS.fold_dir, FLAGS.train_list), FLAGS.data_dir,
                     FLAGS.train_shards)
    
    if len(valid_outcomes) != len(valid_outcomes | train_outcomes):
        print('Warning: unattested labels in training data [%s]' % (', '.join(valid_outcomes | train_outcomes) - valid_outcomes))
        
    output_file = os.path.join(FLAGS.output_dir, 'md.json')


    md = { 'num_valid_shards': FLAGS.valid_shards, 
           'num_train_shards': FLAGS.train_shards,
           'valid_counts': valid,
           'train_counts': train,
           'timestamp': str(datetime.now()),
           'nlabels': len(train_outcomes) }
    with open(output_file, 'w') as f:
        json.dump(md, f)


if __name__ == '__main__':
    tf.app.run()
