import tensorflow as tf
import numpy as np
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_to_tfexample(image_data, label, filepath):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/class/label': bytes_feature(bytes(label, encoding='utf-8')),
        'image/filepath': bytes_feature(bytes(filepath, encoding='utf-8')),
    }))


def get_num_of_files_in_tfrecord(filepath):  # 获取tfrecord中的文件数量
    num = 0
    for record in tf.python_io.tf_record_iterator(filepath):
        num = num + 1
    return num


def parser(record):
    fe = tf.parse_single_example(record, features={
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.int64), })
    return fe


class TFRecord:
    tf_writer = None
    g = tf.Graph()
    sign = True

    def __init__(self):
        with self.g.as_default():
            # tf.reset_default_graph()

            tem_place = tf.placeholder(tf.string, name='filename')
            dataset = tf.data.TFRecordDataset(tem_place)

            dataset = dataset.map(parser)
            # iterator = dataset.make_one_shot_iterator()
            iterator = dataset.make_initializable_iterator()
            features = iterator.get_next()
            # 担心数据不一致，所以转化一次
            image_data = tf.cast(features['image/encoded'], tf.string)
            image_data = tf.image.decode_jpeg(image_data)

            label = tf.cast(features['image/class/label'], tf.int64, name='label')

            tf.add_to_collection('image_data', image_data)
            tf.add_to_collection('iterator', iterator)
            tf.add_to_collection('label', label)

    def image_data_read_jpg_with_dataset_graph_not_load_again(self, filename):
        with tf.Session(graph=self.g) as sess:

            iterator = self.g.get_collection('iterator')[0]
            tem_place = self.g.get_operation_by_name('filename').outputs[0]
            image_data = self.g.get_collection('image_data')[0]
            label = self.g.get_collection('label')[0]

            sess.run(iterator.initializer, feed_dict={tem_place: filename})   #重新初始化读取的文件名
            image_data_list = []
            label_list = []
            filepath_list = []

            num_files = get_num_of_files_in_tfrecord(filename)
            for i in range(num_files):
                [_image_data, _label] = sess.run([image_data, label]
                                                 )
                #_label = str(_label, encoding='utf-8')

                # print(_image_data)
                image_data_list.append(_image_data)
                label_list.append(_label)
        return image_data_list, label_list

    def image_data_writer_open(self, filename):

        self.tf_writer = tf.python_io.TFRecordWriter(filename)

    def image_data_writer_close(self):
        self.tf_writer.close()

    def image_data_write_jpeg(self, image_data, filepath, label='test'):

        g = tf.Graph()
        if (image_data.dtype != np.uint8):
            image_data = image_data.astype(np.uint8)
        with g.as_default():
            image_placeholder = tf.placeholder(dtype=tf.uint8)
            encoded_image = tf.image.encode_jpeg(image_placeholder)  ###为了保证无损压缩,之后可能换成jpeg,对读取来说没有影响

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(graph=g) as sess:
            [_image_encode] = sess.run([encoded_image],
                                       feed_dict={image_placeholder: image_data})
        example = image_to_tfexample(_image_encode, label, filepath)
        self.tf_writer.write(example.SerializeToString())

        return None


if __name__ == '__main__':
    filename = "./dataset/data_flowers/flowers_train_00001-of-00005.tfrecord"

    tf_data = TFRecord()


    st = time.time()

    _image_data, _label = tf_data.image_data_read_jpg_with_dataset_graph_not_load_again(filename)
    end = time.time()
    print(_image_data)

    print(_label)
    print('read time:', end - st)

