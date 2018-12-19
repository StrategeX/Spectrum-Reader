#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Spectrum Reader visualizes .txt files containing spectroscopic data.
    Copyright (C) 2018  Max Asenow

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
    @version 1.4.1
'''

import copy
import math
import os
import re
import sys
import traceback

from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtWidgets import QDialog, QFileDialog,  QInputDialog, QApplication, QPushButton, QVBoxLayout, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from pylatexenc.latex2text import LatexNodes2Text
from scipy.signal import _savitzky_golay as sg
from scipy.signal import argrelmax

import matplotlib.pyplot as plt
import numpy as np


def catch_exceptions(t, val, tb):
    exc = ''.join(traceback.format_exception(t, val, tb))
    QMessageBox.critical(None, "Ein unvorhergesehener Fehler ist aufgetreten",
                         "Fehlertyp: {}\nBeschreibung: {}\n\nDebuginformationen:\n{}".format(t, val, exc))
    old_hook(t, val, tb)


old_hook = sys.excepthook
sys.excepthook = catch_exceptions


class UvVisData():

    def __init__(self, path):
        self.path = path

        with open(path, 'r') as file:
            raw_lines = file.readlines()
            whole_content = ''.join(raw_lines)
            if '\t' in whole_content:
                lines = [line.replace('\n', '').replace(',', '.').split('\t')
                         for line in raw_lines]
            elif ', ' in whole_content:
                lines = [line.replace('\n', '').split(', ')
                         for line in raw_lines]

            elif ';' in whole_content:
                lines = [line.replace('\n', '').replace(',', '.').split(';')
                         for line in raw_lines]

            else:
                # let user input a delimiter
                delimiter, ok = QInputDialog.getText(None, 'Trennzeichen zwischen den Messwerten nicht erkannt.',
                                                           'Bitte geben Sie für {} das Trennzeichen manuell ein.\n("." ist nicht zulässig.)'
                                                           .format(os.path.basename(self.path)))

                if ok and not('.' in delimiter):
                    if ',' in delimiter:
                        lines = [line.replace('\n', '').split(delimiter)
                                 for line in raw_lines]
                    else:
                        lines = [line.replace('\n', '').replace(',', '.').split(delimiter)
                                 for line in raw_lines]

        self.length = len(lines)
        for line_nr in range(0, self.length):
            try:
                float(lines[line_nr][0])
                self.data_start = line_nr
                break
            except:
                # TODO: exception
                pass

        self.npoints = self.length - self.data_start
        self.var_stats = lines[:self.data_start]

        # make sure var_stats contains only pairs
        for pair in self.var_stats:
            if len(pair) < 2:
                pair.append('')
        # Default no header information
        self.title = 'Unbekannt'
        self.date = 'Unbekannt'
        self.time = 'Unbekannt'
        self.mode = 'unbekannte Einheiten'

        if not self.data_start == 0:

            self.title = lines[0][-1]

            # date
            found = False
            for line in lines:
                if re.search('date|Datum| am', line[0], flags=re.IGNORECASE):
                    self.date = line[-1].replace('/', '.')
                    found = True
                    break
            if not found:
                for line in lines:
                    result = re.search('[0-9][0-9][\./][0-9][0-9][\./][0-9][0-9][0-9][0-9]',
                                       ''.join(line))
                    break
                if result:
                    self.date = result.group(0)

            # time
            found = False
            for line in lines:
                if re.search('time|Zeit| um', line[0], flags=re.IGNORECASE):
                    self.time = line[-1]
                    found = True
                    break
            if not found:
                for line in lines:
                    result = re.search('[0-9][0-9]:[0-9][0-9]:[0-9][0-9]',
                                       ''.join(line))
                    break
                if result:
                    self.time = result.group(0)

            # mode
            found = False
            for line in lines:
                if re.search('YUNITS|Modus', line[0], flags=re.IGNORECASE):
                    self.mode = line[-1]
                    found = True
                    break

            if not found:
                self.mode = lines[self.data_start - 1][-1]

        self.wavelength = np.full(self.npoints, -42.0)
        self.y = np.full(self.npoints, -42.0)

        x = 0
        for data in lines[self.data_start:]:
            self.wavelength[x] = float(data[0])
            try:
                self.y[x] = float(data[1])
            except:
                self.y[x] = np.nan
            x += 1

        self.xmin = self.wavelength[0]
        self.xmax = self.wavelength[-1]
        self.ymin = min([el for el in self.y if not math.isnan(el)])
        self.ymax = max([el for el in self.y if not math.isnan(el)])

        try:
            self.deltax = self.wavelength[1] - self.wavelength[0]
        except:
            self.deltax = None

        try:
            self.unit = {'INTENSITY': ['Intensität I', ' in ', 'a.u.'],  # willkürlichen Einheiten #'$W \cdot m^{-2}$'],
                         'A': ['Extinktion E', '', ''],
                         'E': ['Extinktion E', '', ''],
                         '%T': ['Transmission', ' in ', '\\%']}[self.mode]
        except:
            self.unit = ['', '', self.mode]


class Reader(QDialog):
    def __init__(self, parent=None):
        super(Reader, self).__init__(parent)
        self.ui = uic.loadUi('ui_show_uvvis.ui', self)

        # a figure instance to plot on
        self.figure = Figure()  # plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # set the layout
        layout = QVBoxLayout(self.ui.figure_frame)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # data
        self.data = {}

        # show peaks?
        self.show_peaks = False

        # signals
        self.ui.add_reading.clicked.connect(self._add_reading)
        self.ui.remove_reading.clicked.connect(self._remove_reading)
        self.ui.choose_reading.itemSelectionChanged.connect(
            self.update_details)
        self.ui.about.clicked.connect(self._about)
        self.ui.external_window.clicked.connect(self.open_window)
        self.ui.peaks.stateChanged.connect(self.set_peaks)
        self.ui.standardize.stateChanged.connect(self.set_standardize)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowFlags(QtCore.Qt.Window)
        # QtCore.Qt.WindowSystemMenuHint |
        # QtCore.Qt.WindowMinMaxButtonsHint |
        # QtCore.Qt.WindowContextHelpButtonHint)self.windowFlags()

        self.form_list = []
        self.standard = False

        self.last_path = QtCore.QDir.homePath()

    def _add_reading(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Zu importierende Datei wählen', self.last_path)
        try:
            self.last_path = os.path.dirname(paths[0])
        except:
            pass

        for path in paths:
            if path:
                file_name = os.path.basename(path)
                if not(file_name in self.data):
                    try:
                        self.data[file_name] = UvVisData(path)
                        self.ui.choose_reading.addItem(file_name)
                        label = ''.join(self.data[file_name].unit)
                        self.update_figure()
                        # TODO: fix
                    except ZeroDivisionError:
                        QMessageBox.critical(None, 'Bein Einlesen der Datei ist ein Fehler aufgetreten',
                                             '''Möglicherweise ist die Datei beschädigt oder besitzt nicht
                                     das richtige Format.''')
                else:
                    QMessageBox.critical(None, 'Eine Datei mit dem selben Namen wurde bereits importiert!',
                                         'Entfernen sie zuerst die alte Messung oder benennen sie die Datei um.')

    def _remove_reading(self):
        if self.data != {}:
            try:
                file_name = self.ui.choose_reading.currentItem().text()
                del self.data[file_name]
                index = self.ui.choose_reading.row(
                    self.ui.choose_reading.currentItem())
                foo = self.ui.choose_reading.takeItem(index)
                self.update_figure()
            except AttributeError:
                QMessageBox.warning(None, 'Keine Messung zum Entfernen aus dem Diagramm ausgewählt.',
                                    '''Bitte wählen sie im Bereich "Messungen" zunächst eine geöffnete Datei aus.''')

    def update_figure(self):

        # reset figure
        self.figure.clear()

        # check if different y labels exist
        labels = [''.join(readings.unit) for _, readings in self.data.items()]
        uniform = False
        if len(labels) > 0:
            uniform = labels.count(labels[0]) == len(labels)

        # create an axis
        ax = self.figure.add_subplot(111)

        ax.set_xlabel(r'Wellenlänge $\lambda$ in nm')
        if uniform:
            if self.standard:
                ax.set_ylabel('Normierte '
                              + list(self.data.values())[0].unit[0]
                              + ' in %')
            else:
                ax.set_ylabel(labels[0])
        else:

            if self.standard:
                ax.set_ylabel(
                    'Achtung: Unterschiedliche Einheiten. (normiert auf 100 %)')
            elif len(labels) != 0:
                QMessageBox.warning(None, 'Die Einheiten der eingelesenen Dateien unterscheiden sich',
                                    '''Möglicherweise ist eine Vergleichbarkeit der Daten nicht gewährleistet. ''')
                ax.set_ylabel('Achtung: Unterschiedliche Einheiten.')

        # create grid
        ax.grid()

        # plot data
        for file_name, readings in self.data.items():
            yvalues = copy.deepcopy(readings.y)
            if self.standard:
                if readings.ymax != 0:
                    yvalues = (yvalues - readings.ymin) / \
                        (readings.ymax - readings.ymin) * 100
                    # yvalues = yvalues / readings.ymax * 100

            ax.plot(readings.wavelength, yvalues,
                    label=os.path.basename(readings.path).replace('.txt', ''))
            # peaks
            if self.show_peaks:
                smooth_data = sg.savgol_filter(
                    yvalues, window_length=31, polyorder=3)
                maxima = argrelmax(smooth_data)

                for i in np.nditer(maxima):
                    ax.scatter([readings.wavelength[i]], [
                        smooth_data[i]], 10, color='black')
                    ax.plot([readings.wavelength[i], readings.wavelength[i]],
                            [readings.ymin, smooth_data[i]], color='black')
                    if self.standard:
                        peak_label = str(readings.wavelength[i]) + ' nm'
                    else:
                        peak_label = '(' + str(readings.wavelength[i]) + '|' + str(
                            round(smooth_data[i], 2)) + ')'
                    ax.annotate(peak_label,
                                xy=(readings.wavelength[i], yvalues[i]),
                                xytext=(+0, +10), textcoords='offset pixels')
        # legend
        ax.legend()

        # refresh canvas
        self.canvas.draw()

    def update_details(self):
        # update details
        if self.ui.choose_reading.currentItem():
            reading = self.data[self.ui.choose_reading.currentItem().text()]
            self.ui.path.setText(reading.path)
            self.ui.title.setText(reading.title)
            self.ui.mode.setText(reading.mode)
            self.ui.date.setText(reading.date)
            self.ui.time.setText(reading.time)
            self.ui.mrange.setText('{} nm bis {} nm'.format(
                reading.xmin, reading.xmax))
            self.ui.deltax.setText(str(reading.deltax))
            self.ui.min_max.setText('{}/{} {}'.format(reading.ymin,
                                                      reading.ymax,
                                                      self.l2u(reading.unit[2])))

            self.form_list = []
            row = 0

            # clear all previous information
            for i in reversed(range(self.ui.var_details_layout.count())):
                self.ui.var_details_layout.itemAt(i).widget().setParent(None)

            for labeltext, fieldtext in reading.var_stats:

                self.form_list.append((QtWidgets.QLabel(self.ui.var_details_widget),
                                       QtWidgets.QLineEdit(self.ui.var_details_widget)))
                self.form_list[-1][0].setText(labeltext)
                self.ui.var_details_layout.setWidget(
                    row, QtWidgets.QFormLayout.LabelRole, self.form_list[-1][0])

                self.form_list[-1][1].setText(fieldtext)
                self.form_list[-1][1].setReadOnly(True)
                self.ui.var_details_layout.setWidget(
                    row, QtWidgets.QFormLayout.FieldRole, self.form_list[-1][1])
                row += 1
        else:

            # clear all previous information
            for i in reversed(range(self.ui.var_details_layout.count())):
                self.ui.var_details_layout.itemAt(i).widget().setParent(None)

            self.ui.path.setText('')
            self.ui.title.setText('')
            self.ui.mode.setText('')
            self.ui.date.setText('')
            self.ui.time.setText('')
            self.ui.mrange.setText('')
            self.ui.deltax.setText('')
            self.ui.min_max.setText('')

    def _about(self):
        QMessageBox.information(None, 'Über das Progamm',
                                '''
    Für Hilfe klicken sie mit der rechten Maustaste auf das entsprechende Element,
    zu dem Sie Hilfe erhalten wollen.
    
    Version 1.4.1
    
    Spectrum Reader visualizes .txt files containing spectroscopic data.
    Copyright (C) 2018  Max Asenow

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
''')

    def open_window(self):
        back_up_figure = self.figure
        self.figure = plt.figure()
        self.update_figure()
        plt.show()
        self.figure = back_up_figure

    def set_peaks(self):
        self.show_peaks = self.ui.peaks.isChecked()
        self.update_figure()

    def l2u(self, string):
        return LatexNodes2Text().latex_to_text(string)

    def set_standardize(self):
        self.standard = self.ui.standardize.isChecked()
        self.update_figure()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Reader()
    main.show()

    sys.exit(app.exec_())
