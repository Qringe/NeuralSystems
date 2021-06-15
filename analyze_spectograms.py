import numpy as np
import os.path as path
import gc
from joblib import dump, load
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RangeSlider, CheckButtons

from utils import (
    # Constants
    DEVICE, BIRD_NAMES, DATA_PATH, MODEL_PATH, PREDICTIONS_PATH,

    # Data handling functions
    load_bird_data, extract_labelled_spectograms, train_test_split,
    extract_birds, create_windows, store_birds, load_birds, store_dataset,
    load_dataset, flatten_windows_dic, standardize_data,

    # Other stuff
    hash_spectograms, tuples2vector, score_predictions
)

from analyze_errors import analyze_errors


class MyPlot(object):
    '''
    A class used to nicely plot several plots such that the user can switch between plots,
    using a button and a scrollbar.
    '''

    def __init__(self, bird_name, spectograms, predictions, analysis_dic=None, only_errors=False, view_width=1000):
        self.ind = 0
        self.bird_name = bird_name
        self.length = len(spectograms)
        self.spectograms = spectograms
        self.predictions = predictions
        self.view_width = view_width
        self.changed_spectogram = False
        self.show_type1_errors = True
        self.show_type2_errors = True
        self.show_type3_errors = True
        self.show_type4_errors = True
        self.left_border, self.right_border = self.get_slider_init()
        self.analysis_dic = analysis_dic
        self.annotations = []
        self.fig, self.ax = plt.subplots()
        self.plot_spectogram()

    def plot_spectogram(self, artists=["all"]):
        '''
        Plot the spectogram number 'self.ind' together with the predictions, the true syllables
        and the different errors
        '''
        # Local variables used in this function
        message_width = 45
        message_height = 30

        # First clear all syllable-patches and annotations from the previous plot
        for patch in reversed(self.ax.patches):
            patch.remove()
        for an in self.annotations:
            an.remove()
        self.annotations = []

        # Extract the true syllables and the predictions
        tuples_predicted = self.predictions[self.ind]["y_pred"]
        tuples_gt = self.predictions[self.ind]["y_true"]

        # Get the current spectogram and the left and right border of the viewable window
        current_spec = self.spectograms[self.ind]
        _, current_spec_width = current_spec.shape
        left_border_column = round(self.left_border * current_spec_width / 100)
        right_border_column = round(self.right_border * current_spec_width / 100)

        # Only extract the viewable window
        current_spec = current_spec[:, left_border_column:right_border_column]

        # Extract the tuples which contain errors
        errors_type1 = self.analysis_dic["type1_tuples"][self.ind] if self.show_type1_errors else []
        errors_type2 = self.analysis_dic["type2_tuples"][self.ind] if self.show_type2_errors else []
        errors_type3 = self.analysis_dic["type3_tuples"][self.ind] if self.show_type3_errors else []
        errors_type4 = self.analysis_dic["type4_tuples"][self.ind] if self.show_type4_errors else []

        # A list containing all the errors
        errors = []

        # Plot the spectogram
        self.ax.imshow(current_spec)

        # Add the predictions in red
        for t_p in tuples_predicted:
            if t_p[0] > right_border_column or t_p[1] < left_border_column:
                continue

            # Check if this tuple is a type 1 or type 2 error
            error_name = None
            color = 'r'
            if t_p in errors_type2:
                error_name = "Noise\nerror"
            elif t_p in errors_type1:
                error_name = "Border\nerror"
            else:
                color = 'g'

            # Adjust the coordinates of the tuple to the viewable window
            t_p = (max(t_p[0] - left_border_column, 0), min(t_p[1] - left_border_column, right_border_column))

            # If the prediction is erroneous, add it to the error list
            if error_name is not None:
                errors.append((t_p, error_name))

            # Add the predictions patches
            self.ax.add_patch(matplotlib.patches.Rectangle(
                (t_p[0], 0), t_p[1]-t_p[0], 15, linewidth=2,
                edgecolor=color,
                facecolor=color,
                fill=True))

        # Add the true syllables in green
        for t_gt in tuples_gt:
            if t_gt[0] > right_border_column or t_gt[1] < left_border_column:
                continue

            # Check if this tuple is a type 3 error
            error_name = None
            if t_gt in errors_type3:
                error_name = "Skip\nerror"

            # Adjust the coordinates of the tuple to the viewable window
            t_gt = (max(t_gt[0] - left_border_column, 0), min(t_gt[1] - left_border_column, right_border_column))

            # If the prediction is erroneous, add it to the error list
            if error_name is not None:
                errors.append((t_gt, error_name))

            self.ax.add_patch(matplotlib.patches.Rectangle(
                (t_gt[0], 30), t_gt[1]-t_gt[0], 15, linewidth=2,
                edgecolor='y',
                facecolor='y',
                fill=True)
                        )

        # Add all type 4 errors which lie inside the viewable window
        error_name = "Split\nerror"

        for t_e4 in errors_type4:
            if t_e4[0] > right_border_column or t_e4[1] < left_border_column:
                continue

            # Adjust the coordinates of the tuple to the viewable window
            t_e4 = (max(t_e4[0] - left_border_column, 0), min(t_e4[1] - left_border_column, right_border_column))

            errors.append((t_e4, error_name))

        # The following code will ensure that no two error messages will overlap
        # Sort all the errors and initialize helper datastructures
        errors = sorted(errors, key=lambda t: t[0][0])
        indices = [tup[0][0] for tup in errors]
        levels = [0 for i in range(len(errors))]

        error_name_dic = {"Border\nerror": 0, "Noise\nerror": 1, "Skip\nerror": 2, "Split\nerror": 3}
        error_types = [error_name_dic[tup[1]] for tup in errors]

        # Plot all the different error messages
        for i in range(len(errors)):
            # Get the coordinates of this tuple
            left, right = errors[i][0]

            # Get all indices error messages which collide with this one.
            if error_types[i] <= 2:
                conflict_indices = [indices.index(tup[0]) for tup in zip(indices[:i], error_types[:i])
                                    if tup[1] <= 2 and indices[i] - tup[0] <= message_width]
            else:
                conflict_indices = [indices.index(tup[0]) for tup in zip(indices[:i], error_types[:i])
                                    if tup[1] == 3 and indices[i] - tup[0] <= message_width]

            # Get all levels of these error messages
            conflict_levels = {levels[ind] for ind in conflict_indices}

            # Assign this error the smallest level which is not in this list
            levels[i] = min(set(range(len(errors))) - conflict_levels)

        for i in range(len(errors) - 1, -1, -1):
            # Get the coordinates of this tuple
            left, right = errors[i][0]

            # Plot the error
            if error_types[i] <= 1:
                self.annotations.append(self.ax.annotate(
                    errors[i][1], xy=(left + (right - left) / 2, 0), xytext=(0, 10 + levels[i] * message_height),
                    xycoords='data', textcoords='offset points', fontsize=10, ha='center', va='bottom',
                    bbox=dict(boxstyle='square', fc='white'),
                    arrowprops=dict(arrowstyle='-', lw=2.0)
                ))
            elif error_types[i] == 2:
                self.annotations.append(self.ax.annotate(
                    errors[i][1], xy=(left + (right - left) / 2, 30), xytext=(0, 40 + levels[i] * message_height),
                    xycoords='data', textcoords='offset points', fontsize=10, ha='center', va='bottom',
                    bbox=dict(boxstyle='square', fc='white'),
                    arrowprops=dict(arrowstyle='-', lw=2.0)
                ))
            else:
                self.annotations.append(self.ax.annotate(
                    errors[i][1], xy=(left + (right - left) / 2, 10), xytext=(0, -80 - levels[i] * message_height),
                    xycoords='data', textcoords='offset points', fontsize=10, ha='center', va='bottom',
                    bbox=dict(boxstyle='square', fc='white'),
                    arrowprops=dict(arrowstyle='-', lw=2.0)
                ))

        # Finally, set the plot title. The position depends on 'levels' hence the late addition
        levels = levels if len(levels) > 0 else [0]
        self.ax.set_title(
            f"Bird {self.bird_name}, Spectogram {self.ind}",
            fontdict={'fontsize': 30, 'fontweight': 10},
            pad=20 + 30 * (max(levels) + 1))

        gc.collect()
        plt.draw()

    def iterate(self, amount=1):
        def iterate_inner(event):
            '''
            This function is used by the 'next' and 'prev' buttons. It first updates
            the index 'ind', then the scrollbar and finally it plots the 'ind'th spectogram
            '''
            self.ind += amount
            self.ind = self.ind % self.length
            self.changed_spectogram = True
            self.slider.set_val((-1, 101))
            self.plot_spectogram()

        return iterate_inner

    def get_check_boxes(self):
        return (
            self.show_type1_errors,
            self.show_type2_errors,
            self.show_type3_errors,
            self.show_type4_errors)

    def switch_check_boxes(self, label):
        # Get the index of the corresponding boolean variable
        labels = ["Border errors", "Noise errors", "Skip errors", "Split errors"]
        index = labels.index(label)

        # Update the corresponding boolean variable
        if index == 0:
            self.show_type1_errors = not self.show_type1_errors
        elif index == 1:
            self.show_type2_errors = not self.show_type2_errors
        elif index == 2:
            self.show_type3_errors = not self.show_type3_errors
        else:
            self.show_type4_errors = not self.show_type4_errors

        # Update the spectogram view
        self.plot_spectogram(artists=["annotations"])

    def get_slider_init(self):
        # Get the current spectogram
        current_spec = self.spectograms[self.ind]

        # Compute the width of the scrollbar for this spectogram
        _, num_cols = current_spec.shape
        scroll_width = round(100 * min(self.view_width, num_cols) / num_cols)

        # Ensure that the width of the scrollbar is an even number
        if scroll_width % 2 == 1:
            scroll_width += 1

        return (0, scroll_width)

    def build_update_view(self, slider):
        self.slider = slider

        def update_view(pos):
            '''
            This function gets called when the user moves the slider scrollbar. 'pos' is a
            tuple containing the left and right position of the 'RangeSlider'.
            '''
            # Get the desired width of the scrollbar
            init_left, init_right = self.get_slider_init()
            slider_length = init_right - init_left
            half_slider_length = slider_length / 2

            # If the scrollbar already has this width, return
            if pos[1] - pos[0] == slider_length:
                return

            # Otherwise set the new middle of the scrollbar to the point
            # where the user clicked
            elif self.changed_spectogram:
                new_middle = half_slider_length
                self.changed_spectogram = False
            elif pos[0] != self.left_border:
                new_middle = int(pos[0])
            else:
                new_middle = int(pos[1])

            # Ensure that the scrollbar doesn't go over the left and right limits
            if new_middle + half_slider_length > 100:
                new_middle = 100 - half_slider_length
            elif new_middle - half_slider_length < 0:
                new_middle = half_slider_length

            # Set the new scrollbar values by recursively calling this function
            # This recursive call should directly return, using the 'return' statement above
            self.slider.set_val((new_middle - half_slider_length, new_middle + half_slider_length))

            # Update the newly defined left and right borders of the scrollbar
            self.left_border = new_middle - half_slider_length
            self.right_border = new_middle + half_slider_length

            # Update the spectogram view
            self.plot_spectogram()

        return update_view


def plot_predictions(bird_name, spectograms, predictions, analysis_dic=None, only_errors=False):
    # Initialize the plot
    my_plot = MyPlot(bird_name, spectograms, predictions, analysis_dic, only_errors=False)

    # Initialize the buttons
    button_next = Button(plt.axes([0.81, 0.05, 0.1, 0.075]), 'Next')
    button_next.on_clicked(my_plot.iterate(amount=1))
    button_prev = Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Prev')
    button_prev.on_clicked(my_plot.iterate(amount=-1))

    # Initialize the check buttons
    check_buttons = CheckButtons(
        plt.axes([0.7, 0.15, 0.1, 0.1]),
        labels=["Border errors", "Noise errors", "Skip errors", "Split errors"],
        actives=my_plot.get_check_boxes()
        )
    check_buttons.on_clicked(my_plot.switch_check_boxes)

    # Initialize the RangeSlider
    slider = RangeSlider(
        plt.axes([0.10, 0.05, 0.50, 0.03]),
        label="Displayed\nColumns", valmin=0, valmax=100,
        valinit=my_plot.get_slider_init(), valstep=1,
        valfmt='%d %%'
        )
    slider.on_changed(my_plot.build_update_view(slider))

    # Plot the first spectogram
    # plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.show()


if __name__ == "__main__":
    # Choose one of your data files in the predictions folder
    data_name = "predictions_base_CNN_R3428_hash_96e298.data"

    # Automatically extract bird name from 'data_name'
    try:
        bird_name = BIRD_NAMES[[name in data_name for name in BIRD_NAMES].index(True)]
    except ValueError:
        print(("ERROR: The name of the predictions file should contain the name of the bird for "
               f"which the preditions are made. One of {BIRD_NAMES} should be contained in {data_name}"))
        exit()

    # Load the data
    base = path.dirname(path.abspath(__file__))
    p = path.join(base, PREDICTIONS_PATH + data_name)
    try:
        prediction_list = load(p)
    except FileNotFoundError:
        print(f"ERROR: The file '{p}' seems to not exist. Please enter a valid file-name.")
        exit()

    # Analyze the errors of the predictions
    analysis_dic = analyze_errors(prediction_list)

    # Load the spectograms corresponding to this prediction
    print(f"Loading spectograms of bird {bird_name}")
    bird_data = load_bird_data(bird_name)
    bird_data, _ = extract_labelled_spectograms(bird_data)

    print("Plotting predictions")
    plot_predictions(bird_name, bird_data[bird_name]["Xs_train"], prediction_list, analysis_dic)
