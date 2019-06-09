import os
from random import shuffle

import ipywidgets as wg
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, clear_output

from data_loaders.image_loader_labels_from_folders import ImageLoaderFromFolders
from labeler.recommender_system import RecommenderSystem
from labeler.utils import open_files_dialog, save_file_dialog


class Labeler:

    def __init__(self, labels, learner=None):
        self.data_loader = ImageLoaderFromFolders()
        self.labels = labels
        self.learner = learner
        self.index = 0
        self.df = None
        self.df_results = None
        self.disable_model_related_buttons = True

        if self.learner:
            self.recommender_system = RecommenderSystem(labels, learner)
            self.disable_model_related_buttons = False

        self.create_ui_elements()
        self.create_layout()
        self.set_connections()

    def initialize_dataframe(self):
        self.df = pd.DataFrame({'file_name': self.file_names, 'image': self.images})
        self.df['labels'] = None
        indices = self.df.index.tolist()
        shuffle(indices)
        self.samples = iter(indices)
        self.handle_next()

    def create_ui_elements(self):
        self.label_buttons = []
        for label in self.labels:
            button = wg.Button(description=label, disabled=False)
            self.set_button_state(button, False)
            self.label_buttons.append(button)

        self.load_button = wg.Button(description="Load images for labeling")
        self.next_button = wg.Button(description="Next")
        self.save_labels_button = wg.Button(description="Save labels")
        self.save_labeled_images_button = wg.Button(description="Save labeled images")
        self.query_button = wg.Button(description="Query samples", disabled=self.disable_model_related_buttons)
        self.suggest_button = wg.Button(description="Suggest", disabled=self.disable_model_related_buttons)
        self.suggest_button.checked = False

        self.figure, self.axes = plt.subplots(figsize=(14, 6))
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)

    def create_layout(self):
        self.layout = wg.VBox([
            wg.HBox([self.load_button, self.save_labeled_images_button, self.query_button, self.suggest_button]),
            wg.HBox([self.save_labels_button, self.next_button]),
            wg.HBox(self.label_buttons),
        ])

    def set_connections(self):
        for button in self.label_buttons:
            button.on_click(self.toggle_button)
        self.next_button.on_click(self.handle_next)
        self.load_button.on_click(self.handle_load)
        self.save_labels_button.on_click(self.handle_save_labels)
        self.save_labeled_images_button.on_click(self.handle_save_labeled_images)
        self.query_button.on_click(self.handle_query)
        self.suggest_button.on_click(self.handle_suggest)

    def display(self):
        clear_output(wait=True)
        if self.df is not None:
            self.figure, self.axes = plt.subplots(figsize=(14, 6))
            self.axes.get_xaxis().set_visible(False)
            self.axes.get_yaxis().set_visible(False)
            self.axes.imshow(self.df.image.loc[self.index])
        display(self.layout)

    def run(self):
        self.display()

    def toggle_button(self, button):
        button.checked = not button.checked
        self.set_button_color(button)

    def set_button_color(self, button):
        if button.checked:
            button.style.button_color = 'lightgreen'
        else:
            button.style.button_color = 'lightgray'

    def handle_next(self, button=None):
        self.uncheck_label_buttons()
        self.update_index()
        self.make_suggestion()
        self.display()

    def handle_load(self, button=None):
        paths = open_files_dialog()
        if paths:
            self.images, self.paths = self.data_loader._load_images_from_file_paths(paths)
            self.file_names = [os.path.basename(p) for p in self.paths]
            self.initialize_dataframe()

    def handle_save_labels(self, button=None):
        selected_labels = [b.description for b in self.label_buttons if b.checked]
        self.df.loc[self.index, 'labels'] = selected_labels
        self.handle_next()

    def handle_save_labeled_images(self, button=None):
        path = save_file_dialog()
        if path:
            df_labeled = self.df[~self.df.labels.isnull()]
            df_labeled['labels_str'] = df_labeled.labels.apply(lambda x: ', '.join(x))
            df_labeled[['file_name', 'labels_str']].to_csv(path, index=False, header=False)

    def handle_query(self, button=None):
        samples, confidences, predicted_labels = self.recommender_system.query(self.df)
        self.df_results = pd.DataFrame({'sample': samples,
                                        'confidence': confidences,
                                        'predicted_label': predicted_labels})
        self.df_results = self.df_results.set_index('sample')

        self.samples = iter(samples)
        self.handle_next()

    def handle_suggest(self, button):
        self.toggle_button(button)

    def update_index(self):
        try:
            self.index = next(self.samples)
        except StopIteration:
            pass

    def make_suggestion(self):
        if self.df_results is not None and self.suggest_button.checked:
            labels = self.df_results.loc[self.index].predicted_label
            self.check_label_buttons(labels)

    def uncheck_label_buttons(self):
        for button in self.label_buttons:
            self.set_button_state(button, False)

    def check_label_buttons(self, labels):
        for button in self.label_buttons:
            if button.description in labels:
                self.set_button_state(button, True)

    def set_button_state(self, button, is_checked):
        button.checked = is_checked
        self.set_button_color(button)
