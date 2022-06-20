import os
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import pywt
from torch.utils.data import Dataset
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


class Segment:
    def __init__(self, data, interval):
        """
        param data:
        param interval:
        """
        self.data = data
        self.interval = interval

        self.start_index = self.interval[0]
        self.end_index = self.interval[1]

        expected_class = self.data[0, -1]
        mean_class = np.mean(self.data[:, -1])
        assert expected_class == mean_class, f"Average class {mean_class} is not equal to {expected_class}"
        self.class_number = int(expected_class)

        self.time_interval = self.data[-1, 0] - self.data[0, 0]
        self.length = self.data.shape[0]
        self.channels = self.data.shape[1] - 2

        # self.normalization_info = normalization_info


class SingleSubject:

    def __init__(self, root_dir, subject_id, file_name, channel_avg_remove=False):
        """
        param root_dir:
        param subject_id:
        param file_name:
        """
        self.subject_id = subject_id
        self.file_name = file_name
        self.file_path = os.path.join(root_dir, subject_id, file_name)

        self.dataframe = pd.read_csv(self.file_path, sep="\t")
        self.dataframe.fillna(0, inplace=True)

        raw_data = self.dataframe.values

        if channel_avg_remove:
            channel_avg = np.mean(raw_data[:, 1:-1], axis=1)
            channel_avg = np.expand_dims(channel_avg, axis=1)
            channel_avg = np.repeat(channel_avg, raw_data.shape[1] - 2, axis=1)
            raw_data[:, 1:-1] = raw_data[:, 1:-1] - channel_avg

        self.segments = []
        segment_margins = np.nonzero(np.diff(raw_data[:, -1]))[0]

        for index, margin in enumerate(segment_margins):

            if index == 0:
                segment_start_point = 0
                segment_end_point = margin + 1

                segment_interval = segment_start_point, segment_end_point
                self.segments.append(
                    Segment(data=raw_data[segment_start_point:segment_end_point, :], interval=segment_interval))

            if index != (len(segment_margins) - 1):

                segment_start_point = margin + 1
                segment_end_point = segment_margins[index + 1] + 1

            else:

                segment_start_point = margin + 1
                segment_end_point = raw_data.shape[0]

            segment_interval = segment_start_point, segment_end_point
            self.segments.append(
                Segment(data=raw_data[segment_start_point:segment_end_point, :], interval=segment_interval))

        self.normalization_info = dict()

        linear_scaling = dict()
        linear_scaling['offset'] = raw_data.min(axis=0)[1:-1]
        linear_scaling['scale'] = (raw_data.max(axis=0)[1:-1] - raw_data.min(axis=0)[1:-1])
        self.normalization_info['linear_scaling'] = linear_scaling

        z_score = dict()
        z_score['offset'] = raw_data.mean(axis=0)[1:-1]
        z_score['scale'] = np.sqrt(raw_data.var(axis=0)[1:-1])
        self.normalization_info['z_score'] = z_score

        temp = self.get_segments([1])
        rest_data = temp[0].data
        for i in range(1, len(temp)):
            rest_data = np.concatenate((rest_data, temp[i].data), axis=0)

        rest_z_score = dict()
        rest_z_score['offset'] = rest_data.mean(axis=0)[1:-1]
        rest_z_score['scale'] = np.sqrt(rest_data.var(axis=0)[1:-1])
        self.normalization_info['rest_z_score'] = rest_z_score

        self.dataframe['subject'] = np.asarray([int(subject_id)] * len(self.dataframe))

        experiment = int(self.file_name.split('_')[0])

        self.dataframe['experiment'] = np.asarray([experiment] * len(self.dataframe))

    def get_segments(self, selected_classes=None):
        """
        param selected_classes: list of labels to filter from all segments, None means all classes
        return: the list of selected segment
        """
        selected_segments = []
        for segment in self.segments:
            if selected_classes is not None:
                if segment.class_number in selected_classes:
                    selected_segments.append(segment)
            else:
                selected_segments.append(segment)
        return selected_segments


class EmgGestures:

    def __init__(self, root_dir, channel_avg_remove=False):
        """
        param root_dir: root director of data
        """
        self.root_dir = root_dir
        self.ids = [name for name in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, name))]
        self.subjects = []
        self.dataframe = None

        for id_folder in self.ids:
            folder_path = join(root_dir, id_folder)
            files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
            for dataset_file in files:
                temp_subject = SingleSubject(root_dir, id_folder, dataset_file, channel_avg_remove=channel_avg_remove)
                self.subjects.append(temp_subject)
                if self.dataframe is None:
                    self.dataframe = temp_subject.dataframe
                else:
                    self.dataframe = pd.concat([self.dataframe, temp_subject.dataframe], ignore_index=True)
                    # self.dataframe = self.dataframe.append(temp_subject.dataframe, ignore_index=True)

        self.dataframe.drop('time', inplace=True, axis=1)

    def get_segments(self, selected_classes=None, alignment='zero_pad', return_time=False,
                     normalization='linear_scaling'
                     ):
        """
        param selected_classes: list of labels to filter from all segments, None means all classes

        param alignment: how to align segment data:
                'zero_pad' -> to maximum length
                'crop' -> to minimum length
                'average_pad' -> pad with mean of data

        param normalization:
                'None' -> No Normalization
                'linear_scaling' -> Linear Scaling Normalization (norm = (x - x_min) / (x_max - x_min))
                'z_score' -> Z-Score Normalization norm = (x - x_mean) / x_var
                'rest_z_score' -> Z-Score in rest Time norm = (x - mean_rest) / var_rest

        return: x: a numpy array with the shape (number of segments, data length, channels)

                y: a numpy array with the shape (number of segments) which indicates the labels

                t: a numpy array with the shape (number of segments, data length) (when return_time == True)
        """
        selected_segments = []
        subjects_ids = []

        for subject in self.subjects:
            subject_segments = subject.get_segments(selected_classes)
            subjects_ids += [int(subject.subject_id)] * len(subject_segments)

            if normalization is not None:
                normalization_info = subject.normalization_info[normalization]

                for segment in subject_segments:
                    segment.data[:, 1:-1] = (segment.data[:, 1:-1] - normalization_info['offset']) / normalization_info[
                        'scale']

            selected_segments += subject_segments

        segment_length = []
        for segment in selected_segments:
            segment_length.append(segment.length)

        number_of_segments = len(selected_segments)

        data_length = 0
        if alignment == 'zero_pad':
            data_length = max(segment_length)
        elif alignment == 'crop':
            data_length = min(segment_length)

        channels = selected_segments[0].channels

        x = np.zeros((number_of_segments, data_length, channels))

        y = np.zeros(number_of_segments)

        t = np.zeros((number_of_segments, data_length))

        s = np.asarray(subjects_ids)

        for idx, segment in enumerate(selected_segments):
            last_index = min(data_length, segment.length)
            x[idx, 0:last_index, :] = segment.data[0:last_index, 1:-1]
            y[idx] = segment.class_number
            t[idx, 0:last_index] = segment.data[0:last_index, 0]

        if return_time:
            return x, y, s, t
        else:
            return x, y, s

    def get_features(self, feature_list, remove_channel_mean=False, include_channel_mean=False):

        dataframe = self.dataframe.copy()

        if remove_channel_mean or include_channel_mean:

            channels_mean = np.mean(dataframe.iloc[:, :8], axis=1)

            channels_mean = np.expand_dims(channels_mean, -1)

            channels_mean = np.repeat(channels_mean, 8, -1)

            if remove_channel_mean:
                dataframe.iloc[:, :8] -= channels_mean

            if include_channel_mean:
                dataframe['channel_mean'] = channels_mean[:, 0]

        dataframe = dataframe.groupby(['subject', 'experiment', 'class'])

        features = dataframe.agg(feature_list)

        features = features.reset_index()

        return features


def create_wavelet_dataset(root_dir, selected_classes, train_test='test', window_size=50, stride=8, wavelet_width=30,
                           down_sample=2):
    emd_dataset = EmgGestures(root_dir=root_dir)
    x, y, s = emd_dataset.get_segments(selected_classes=selected_classes, alignment='crop')

    if train_test == 'train':
        x = x[s < 24]
        y = y[s < 24]
    else:
        x = x[s >= 24]
        y = y[s >= 24]

    widths = np.arange(1, wavelet_width + 1)
    index = 0

    agg = np.zeros(((1000 // stride * x.shape[0]), wavelet_width // down_sample, window_size // down_sample, 8))
    labels = np.zeros((agg.shape[0]))

    for segment_n in tqdm(range(x.shape[0])):
        for window in range(0, 1000, stride):
            for channel in range(8):
                sig = x[segment_n, window:window + window_size, channel]
                sig_wvl = signal.cwt(sig, signal.ricker, widths)
                sig_wvl = sig_wvl[::down_sample, ::down_sample]
                agg[index, :, :, channel] = sig_wvl
            labels[index] = int(y[segment_n])
            index += 1

    print(index)
    print(agg.shape)
    agg = agg.astype(np.float)
    labels = labels.astype(np.uint8)

    np.savez_compressed(f'dataset_{train_test}.npy', data=agg, labels=labels)


if __name__ == '__main__':

    # links = {"a": ["aa", "ab", "ac"], "b": ["ba", "bb", "bc", "bd"]}
    #
    # records = [(k, i) for k, v in links.items() for i in v]
    # df = pd.DataFrame.from_records(records, columns=[f"ch{i}" for i in range(8)])

    emd_dataset = EmgGestures(root_dir='./EMG_data_for_gestures-master')
    x, y, s = emd_dataset.get_segments(selected_classes=[1, 2, 3, 4, 5, 6], alignment='crop')

    records = [(pd.Series(x[i, :, 0]), pd.Series(x[i, :, 1]), pd.Series(x[i, :, 2]), pd.Series(x[i, :, 3]),
                pd.Series(x[i, :, 4]), pd.Series(x[i, :, 5]), pd.Series(x[i, :, 6]), pd.Series(x[i, :, 7])) for i in
               range(x.shape[0])]

    df = pd.DataFrame.from_records(records, columns=[f"ch{i}" for i in range(8)])

    index_list = []

    # for i in range(x.shape[0]):
    #     index_list += ([i] * x.shape[1])
    #
    # index_list = np.asarray(index_list)
    #
    # df = pd.DataFrame(index=index_list, data=x.reshape(-1, x.shape[-1]))

    df.to_pickle('/home/danial/Documents/Projects/mvts_transformer/src/data/EmgGesture/EmgGesture.pkl')

    np.savez_compressed('/home/danial/Documents/Projects/mvts_transformer/src/data/EmgGesture/EmgGesture', data=y)

    print("Hello")

    # create_wavelet_dataset('./EMG_data_for_gestures-master',
    #                        selected_classes=[1, 2, 3, 4, 5, 6])
