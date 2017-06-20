"""
The package assumes the following project layout to make the interface lean:

data/
    aux/
    preprocessed/
    features/
        X_train_%featurelistname%.names
        X_train_%featurelistname%.pickle
        X_test_%featurelistname%.pickle
        ...
    submissions/
        %leaderboardscore%_%date%_%description%_%cvscore%.csv
        ...
    trained/
    tmp/
    ...

    train.csv
    test.csv
    ...

notebooks/
...

"""

# TODO: Add submission management.

import os

import numpy as np
import pandas as pd

from .io import load, load_lines, save, save_lines


class Project:
    def __init__(self, root_dir):
        self._root_dir = root_dir
        self._compute_dependent_paths()

    def _compute_dependent_paths(self):
        self._data_dir = os.path.join(self._root_dir, 'data')
        self._notebooks_dir = os.path.join(self._root_dir, 'notebooks')
        self._aux_data_dir = os.path.join(self._data_dir, 'aux')
        self._preprocessed_data_dir = os.path.join(self._data_dir, 'preprocessed')
        self._features_dir = os.path.join(self._data_dir, 'features')
        self._submissions_dir = os.path.join(self._data_dir, 'submissions')
        self._trained_model_dir = os.path.join(self._data_dir, 'trained')
        self._temp_dir = os.path.join(self._data_dir, 'tmp')

    @property
    def root_dir(self):
        return self._root_dir + os.path.sep

    @property
    def data_dir(self):
        return self._data_dir + os.path.sep

    @property
    def notebooks_dir(self):
        return self._notebooks_dir + os.path.sep

    @property
    def aux_dir(self):
        return self._aux_data_dir + os.path.sep

    @property
    def preprocessed_data_dir(self):
        return self._preprocessed_data_dir + os.path.sep

    @property
    def features_dir(self):
        return self._features_dir + os.path.sep

    @property
    def submissions_dir(self):
        return self._submissions_dir + os.path.sep

    @property
    def trained_model_dir(self):
        return self._trained_model_dir + os.path.sep

    @property
    def temp_dir(self):
        return self._temp_dir + os.path.sep

    def load_feature_lists(self, feature_lists):
        """
        Load pickled features for train and test sets, assuming they are saved
        in the `features` folder along with their column names.

        Args:
            feature_lists: A list containing the names of the feature lists to load.

        Returns:
            A tuple containing 3 items: train dataframe, test dataframe,
            and a list describing the index ranges for the feature lists.
        """

        column_names = []
        feature_ranges = []
        running_feature_count = 0

        for list_id in feature_lists:
            feature_list_names = load_lines(self.features_dir + 'X_train_{}.names'.format(list_id))
            column_names.extend(feature_list_names)
            start_index = running_feature_count
            end_index = running_feature_count + len(feature_list_names) - 1
            running_feature_count += len(feature_list_names)
            feature_ranges.append([list_id, start_index, end_index])

        X_train = np.hstack([
            load(self.features_dir + 'X_train_{}.pickle'.format(list_id))
            for list_id in feature_lists
        ])
        X_test = np.hstack([
            load(self.features_dir + 'X_test_{}.pickle'.format(list_id))
            for list_id in feature_lists
        ])

        df_train = pd.DataFrame(X_train, columns=column_names)
        df_test = pd.DataFrame(X_test, columns=column_names)

        return df_train, df_test, feature_ranges

    def save_features(self, train_features, test_features, feature_names, feature_list_id):
        """
        Save features for the training and test sets to disk, along with their metadata.

        Args:
            train_features: A NumPy array of features for the training set.
            test_features: A NumPy array of features for the test set.
            feature_names: A list containing the names of the feature columns.
            feature_list_id: The name for this feature list.
        """

        self.save_feature_names(feature_names, feature_list_id)
        self.save_feature_list(train_features, 'train', feature_list_id)
        self.save_feature_list(test_features, 'test', feature_list_id)

    def save_feature_names(self, feature_names, feature_list_id):
        """
        Save the names of the features for the given feature list to a metadata file.
        Example: `save_feature_names(['num_employees', 'stock_price'], 'company')`.

        Args:
            feature_names: A list containing the names of the features, matching the column order.
            feature_list_id: The name for this feature list.
        """

        save_lines(feature_names, self.features_dir + 'X_train_{}.names'.format(feature_list_id))

    def save_feature_list(self, obj, set_id, feature_list_id):
        """
        Pickle the specified feature list to a file.
        Example: `save_feature_list(project, X_tfidf_train, 'train', 'tfidf')`.

        Args:
            obj: The object to pickle (e.g., a numpy array or a Pandas dataframe)
            project: An instance of pygoose project.
            set_id: The id of the subset (e.g., 'train' or 'test')
            feature_list_id: The name for this feature list.
        """

        save(obj, self.features_dir + 'X_{}_{}.pickle'.format(set_id, feature_list_id))

    @staticmethod
    def discover():
        """
        Automatically discover the paths to various data folders in this project
        and compose a Project instance.

        Returns:
            A constructed Project object.

        Raises:
            ValueError: if the paths could not be figured out automatically.
                In this case, you have to create a Project manually using the initializer.
        """

        # Try ../data: we're most likely running a Jupyter notebook from the 'notebooks' directory
        candidate_path = os.path.abspath(os.path.join(os.curdir, os.pardir, 'data'))
        if os.path.exists(candidate_path):
            return Project(os.path.abspath(os.path.join(candidate_path, os.pardir)))

        # Try ./data
        candidate_path = os.path.abspath(os.path.join(os.curdir, 'data'))
        if os.path.exists(candidate_path):
            return Project(os.path.abspath(os.curdir))

        # Try ../../data
        candidate_path = os.path.abspath(os.path.join(os.curdir, os.pardir, 'data'))
        if os.path.exists(candidate_path):
            return Project(os.path.abspath(os.path.join(candidate_path, os.pardir, os.pardir)))

        # Out of ideas at this point.
        raise ValueError('Cannot discover the structure of the project. Make sure that the data directory exists')

    @staticmethod
    def init():
        """
        Creates the project infrastructure assuming the current directory is the project root.
        Typically used as a command-line entry point called by `pygoose init`.
        """

        project = Project(os.path.abspath(os.getcwd()))
        paths_to_create = [
            project.data_dir,
            project.notebooks_dir,
            project.aux_dir,
            project.features_dir,
            project.preprocessed_data_dir,
            project.submissions_dir,
            project.trained_model_dir,
            project.temp_dir,
        ]

        for path in paths_to_create:
            os.makedirs(path, exist_ok=True)
