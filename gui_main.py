# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\NN\\gui_main.ui'
#
# Created by: PyQt5 UI code generator 5.10
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(960, 540)
        MainWindow.setMinimumSize(QtCore.QSize(960, 540))
        MainWindow.setMaximumSize(QtCore.QSize(960, 540))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 960, 500))
        self.tabWidget.setAcceptDrops(False)
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setElideMode(QtCore.Qt.ElideNone)
        self.tabWidget.setObjectName("tabWidget")
        self.tabSpectrums = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabSpectrums.sizePolicy().hasHeightForWidth())
        self.tabSpectrums.setSizePolicy(sizePolicy)
        self.tabSpectrums.setAutoFillBackground(True)
        self.tabSpectrums.setObjectName("tabSpectrums")
        self.gridLayoutWidget = QtWidgets.QWidget(self.tabSpectrums)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 60, 951, 411))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.plot_spec_W1 = QtWidgets.QWidget(self.gridLayoutWidget)
        self.plot_spec_W1.setObjectName("plot_spec_W1")
        self.gridLayout.addWidget(self.plot_spec_W1, 0, 0, 1, 1)
        self.plot_spec_W2 = QtWidgets.QWidget(self.gridLayoutWidget)
        self.plot_spec_W2.setObjectName("plot_spec_W2")
        self.gridLayout.addWidget(self.plot_spec_W2, 0, 1, 1, 1)
        self.plot_spec_S1 = QtWidgets.QWidget(self.gridLayoutWidget)
        self.plot_spec_S1.setObjectName("plot_spec_S1")
        self.gridLayout.addWidget(self.plot_spec_S1, 1, 0, 1, 1)
        self.plot_spec_S2 = QtWidgets.QWidget(self.gridLayoutWidget)
        self.plot_spec_S2.setObjectName("plot_spec_S2")
        self.gridLayout.addWidget(self.plot_spec_S2, 1, 1, 1, 1)
        self.BT_spec_play = QtWidgets.QPushButton(self.tabSpectrums)
        self.BT_spec_play.setEnabled(False)
        self.BT_spec_play.setGeometry(QtCore.QRect(390, 30, 75, 23))
        self.BT_spec_play.setObjectName("BT_spec_play")
        self.BT_spec_stop = QtWidgets.QPushButton(self.tabSpectrums)
        self.BT_spec_stop.setEnabled(False)
        self.BT_spec_stop.setGeometry(QtCore.QRect(490, 30, 75, 23))
        self.BT_spec_stop.setObjectName("BT_spec_stop")
        self.LB_spec_time = QtWidgets.QLabel(self.tabSpectrums)
        self.LB_spec_time.setEnabled(False)
        self.LB_spec_time.setGeometry(QtCore.QRect(580, 30, 161, 21))
        self.LB_spec_time.setObjectName("LB_spec_time")
        self.CB_spec_sound1 = QtWidgets.QComboBox(self.tabSpectrums)
        self.CB_spec_sound1.setEnabled(False)
        self.CB_spec_sound1.setGeometry(QtCore.QRect(90, 30, 91, 22))
        self.CB_spec_sound1.setObjectName("CB_spec_sound1")
        self.CB_spec_sound1.addItem("")
        self.CB_spec_sound1.addItem("")
        self.CB_spec_sound1.addItem("")
        self.CB_spec_sound1.addItem("")
        self.CB_spec_sound2 = QtWidgets.QComboBox(self.tabSpectrums)
        self.CB_spec_sound2.setEnabled(False)
        self.CB_spec_sound2.setGeometry(QtCore.QRect(770, 30, 91, 22))
        self.CB_spec_sound2.setObjectName("CB_spec_sound2")
        self.CB_spec_sound2.addItem("")
        self.CB_spec_sound2.addItem("")
        self.CB_spec_sound2.addItem("")
        self.CB_spec_sound2.addItem("")
        self.LE_spec_file1 = QtWidgets.QLineEdit(self.tabSpectrums)
        self.LE_spec_file1.setGeometry(QtCore.QRect(10, 0, 431, 20))
        self.LE_spec_file1.setText("")
        self.LE_spec_file1.setObjectName("LE_spec_file1")
        self.TB_spec_file1 = QtWidgets.QToolButton(self.tabSpectrums)
        self.TB_spec_file1.setGeometry(QtCore.QRect(440, 0, 25, 19))
        self.TB_spec_file1.setObjectName("TB_spec_file1")
        self.LE_spec_file2 = QtWidgets.QLineEdit(self.tabSpectrums)
        self.LE_spec_file2.setGeometry(QtCore.QRect(480, 0, 431, 20))
        self.LE_spec_file2.setObjectName("LE_spec_file2")
        self.TB_spec_file2 = QtWidgets.QToolButton(self.tabSpectrums)
        self.TB_spec_file2.setGeometry(QtCore.QRect(910, 0, 25, 19))
        self.TB_spec_file2.setObjectName("TB_spec_file2")
        self.BT_spec_open1 = QtWidgets.QPushButton(self.tabSpectrums)
        self.BT_spec_open1.setGeometry(QtCore.QRect(10, 30, 75, 23))
        self.BT_spec_open1.setObjectName("BT_spec_open1")
        self.BT_spec_open2 = QtWidgets.QPushButton(self.tabSpectrums)
        self.BT_spec_open2.setGeometry(QtCore.QRect(870, 30, 75, 23))
        self.BT_spec_open2.setObjectName("BT_spec_open2")
        self.BT_spec_loadFiles = QtWidgets.QPushButton(self.tabSpectrums)
        self.BT_spec_loadFiles.setEnabled(False)
        self.BT_spec_loadFiles.setGeometry(QtCore.QRect(290, 30, 75, 23))
        self.BT_spec_loadFiles.setObjectName("BT_spec_loadFiles")
        self.tabWidget.addTab(self.tabSpectrums, "")
        self.tabConvertSpec = QtWidgets.QWidget()
        self.tabConvertSpec.setAutoFillBackground(True)
        self.tabConvertSpec.setObjectName("tabConvertSpec")
        self.TB_conv_files = QtWidgets.QToolButton(self.tabConvertSpec)
        self.TB_conv_files.setGeometry(QtCore.QRect(10, 210, 181, 19))
        self.TB_conv_files.setObjectName("TB_conv_files")
        self.GB_conv_noise = QtWidgets.QGroupBox(self.tabConvertSpec)
        self.GB_conv_noise.setGeometry(QtCore.QRect(210, 10, 341, 241))
        self.GB_conv_noise.setObjectName("GB_conv_noise")
        self.formLayoutWidget = QtWidgets.QWidget(self.GB_conv_noise)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 20, 321, 211))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.LB_conv_mulv = QtWidgets.QLabel(self.formLayoutWidget)
        self.LB_conv_mulv.setObjectName("LB_conv_mulv")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.LB_conv_mulv)
        self.CB_conv_mulv = QtWidgets.QComboBox(self.formLayoutWidget)
        self.CB_conv_mulv.setObjectName("CB_conv_mulv")
        self.CB_conv_mulv.addItem("")
        self.CB_conv_mulv.addItem("")
        self.CB_conv_mulv.addItem("")
        self.CB_conv_mulv.addItem("")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.CB_conv_mulv)
        self.LB_conv_noise1 = QtWidgets.QLabel(self.formLayoutWidget)
        self.LB_conv_noise1.setObjectName("LB_conv_noise1")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.LB_conv_noise1)
        self.SB_conv_noise1 = QtWidgets.QSpinBox(self.formLayoutWidget)
        self.SB_conv_noise1.setEnabled(False)
        self.SB_conv_noise1.setMinimum(1)
        self.SB_conv_noise1.setMaximum(255)
        self.SB_conv_noise1.setProperty("value", 1)
        self.SB_conv_noise1.setObjectName("SB_conv_noise1")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.SB_conv_noise1)
        self.LB_conv_noise2 = QtWidgets.QLabel(self.formLayoutWidget)
        self.LB_conv_noise2.setObjectName("LB_conv_noise2")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.LB_conv_noise2)
        self.SB_conv_noise2 = QtWidgets.QSpinBox(self.formLayoutWidget)
        self.SB_conv_noise2.setEnabled(False)
        self.SB_conv_noise2.setMinimum(1)
        self.SB_conv_noise2.setMaximum(255)
        self.SB_conv_noise2.setProperty("value", 1)
        self.SB_conv_noise2.setObjectName("SB_conv_noise2")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.SB_conv_noise2)
        self.LB_conv_noise3 = QtWidgets.QLabel(self.formLayoutWidget)
        self.LB_conv_noise3.setObjectName("LB_conv_noise3")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.LB_conv_noise3)
        self.SB_conv_noise3 = QtWidgets.QSpinBox(self.formLayoutWidget)
        self.SB_conv_noise3.setEnabled(False)
        self.SB_conv_noise3.setMinimum(1)
        self.SB_conv_noise3.setMaximum(255)
        self.SB_conv_noise3.setProperty("value", 1)
        self.SB_conv_noise3.setObjectName("SB_conv_noise3")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.SB_conv_noise3)
        self.LB_conv_noise4 = QtWidgets.QLabel(self.formLayoutWidget)
        self.LB_conv_noise4.setObjectName("LB_conv_noise4")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.LB_conv_noise4)
        self.SB_conv_noise4 = QtWidgets.QSpinBox(self.formLayoutWidget)
        self.SB_conv_noise4.setEnabled(False)
        self.SB_conv_noise4.setMinimum(1)
        self.SB_conv_noise4.setMaximum(255)
        self.SB_conv_noise4.setProperty("value", 1)
        self.SB_conv_noise4.setObjectName("SB_conv_noise4")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.SB_conv_noise4)
        self.LB_conv_noise5 = QtWidgets.QLabel(self.formLayoutWidget)
        self.LB_conv_noise5.setObjectName("LB_conv_noise5")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.LB_conv_noise5)
        self.SB_conv_noise5 = QtWidgets.QSpinBox(self.formLayoutWidget)
        self.SB_conv_noise5.setEnabled(False)
        self.SB_conv_noise5.setMinimum(1)
        self.SB_conv_noise5.setMaximum(255)
        self.SB_conv_noise5.setProperty("value", 1)
        self.SB_conv_noise5.setObjectName("SB_conv_noise5")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.SB_conv_noise5)
        self.LB_conv_noise6 = QtWidgets.QLabel(self.formLayoutWidget)
        self.LB_conv_noise6.setObjectName("LB_conv_noise6")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.LB_conv_noise6)
        self.SB_conv_noise6 = QtWidgets.QSpinBox(self.formLayoutWidget)
        self.SB_conv_noise6.setEnabled(False)
        self.SB_conv_noise6.setMinimum(1)
        self.SB_conv_noise6.setMaximum(255)
        self.SB_conv_noise6.setProperty("value", 1)
        self.SB_conv_noise6.setObjectName("SB_conv_noise6")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.SB_conv_noise6)
        self.LB_conv_noise7 = QtWidgets.QLabel(self.formLayoutWidget)
        self.LB_conv_noise7.setObjectName("LB_conv_noise7")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.LB_conv_noise7)
        self.SB_conv_noise7 = QtWidgets.QSpinBox(self.formLayoutWidget)
        self.SB_conv_noise7.setEnabled(False)
        self.SB_conv_noise7.setMinimum(1)
        self.SB_conv_noise7.setMaximum(255)
        self.SB_conv_noise7.setProperty("value", 1)
        self.SB_conv_noise7.setObjectName("SB_conv_noise7")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.SB_conv_noise7)
        self.PTE_conv_files = QtWidgets.QPlainTextEdit(self.tabConvertSpec)
        self.PTE_conv_files.setGeometry(QtCore.QRect(10, 10, 181, 191))
        self.PTE_conv_files.setObjectName("PTE_conv_files")
        self.LE_conv_foldersave = QtWidgets.QLineEdit(self.tabConvertSpec)
        self.LE_conv_foldersave.setEnabled(False)
        self.LE_conv_foldersave.setGeometry(QtCore.QRect(570, 120, 181, 20))
        self.LE_conv_foldersave.setObjectName("LE_conv_foldersave")
        self.TB_conv_folder = QtWidgets.QToolButton(self.tabConvertSpec)
        self.TB_conv_folder.setEnabled(False)
        self.TB_conv_folder.setGeometry(QtCore.QRect(750, 120, 25, 19))
        self.TB_conv_folder.setObjectName("TB_conv_folder")
        self.BT_conv_load = QtWidgets.QPushButton(self.tabConvertSpec)
        self.BT_conv_load.setEnabled(True)
        self.BT_conv_load.setGeometry(QtCore.QRect(10, 240, 181, 23))
        self.BT_conv_load.setObjectName("BT_conv_load")
        self.BT_conv_convert = QtWidgets.QPushButton(self.tabConvertSpec)
        self.BT_conv_convert.setEnabled(False)
        self.BT_conv_convert.setGeometry(QtCore.QRect(570, 10, 181, 23))
        self.BT_conv_convert.setObjectName("BT_conv_convert")
        self.progressBar_conv = QtWidgets.QProgressBar(self.tabConvertSpec)
        self.progressBar_conv.setEnabled(False)
        self.progressBar_conv.setGeometry(QtCore.QRect(570, 40, 181, 23))
        self.progressBar_conv.setProperty("value", 0)
        self.progressBar_conv.setObjectName("progressBar_conv")
        self.BT_conv_save = QtWidgets.QPushButton(self.tabConvertSpec)
        self.BT_conv_save.setEnabled(False)
        self.BT_conv_save.setGeometry(QtCore.QRect(570, 150, 181, 23))
        self.BT_conv_save.setObjectName("BT_conv_save")
        self.CB_conv_steps = QtWidgets.QComboBox(self.tabConvertSpec)
        self.CB_conv_steps.setGeometry(QtCore.QRect(360, 260, 181, 22))
        self.CB_conv_steps.setMaxVisibleItems(12)
        self.CB_conv_steps.setObjectName("CB_conv_steps")
        self.CB_conv_steps.addItem("")
        self.CB_conv_steps.addItem("")
        self.CB_conv_steps.addItem("")
        self.CB_conv_steps.addItem("")
        self.CB_conv_steps.addItem("")
        self.CB_conv_steps.addItem("")
        self.CB_conv_steps.addItem("")
        self.CB_conv_steps.addItem("")
        self.CB_conv_steps.addItem("")
        self.CB_conv_steps.addItem("")
        self.CB_conv_steps.addItem("")
        self.CB_conv_steps.addItem("")
        self.LB_conv_steps = QtWidgets.QLabel(self.tabConvertSpec)
        self.LB_conv_steps.setGeometry(QtCore.QRect(330, 260, 21, 22))
        self.LB_conv_steps.setObjectName("LB_conv_steps")
        self.tabWidget.addTab(self.tabConvertSpec, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 960, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.CB_spec_sound1.setCurrentIndex(2)
        self.CB_spec_sound2.setCurrentIndex(3)
        self.CB_conv_steps.setCurrentIndex(11)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "???????????????????????????? ???????????? v0.1(pre-alpha) ?? ???????????????? ??.??."))
        self.BT_spec_play.setText(_translate("MainWindow", "play"))
        self.BT_spec_stop.setText(_translate("MainWindow", "stop"))
        self.LB_spec_time.setText(_translate("MainWindow", "time: SS.mmmmm/SS.mmmmm"))
        self.CB_spec_sound1.setCurrentText(_translate("MainWindow", "Left channel"))
        self.CB_spec_sound1.setItemText(0, _translate("MainWindow", "No sound"))
        self.CB_spec_sound1.setItemText(1, _translate("MainWindow", "Both channels"))
        self.CB_spec_sound1.setItemText(2, _translate("MainWindow", "Left channel"))
        self.CB_spec_sound1.setItemText(3, _translate("MainWindow", "Right channel"))
        self.CB_spec_sound2.setCurrentText(_translate("MainWindow", "Right channel"))
        self.CB_spec_sound2.setItemText(0, _translate("MainWindow", "No sound"))
        self.CB_spec_sound2.setItemText(1, _translate("MainWindow", "Both channels"))
        self.CB_spec_sound2.setItemText(2, _translate("MainWindow", "Left channel"))
        self.CB_spec_sound2.setItemText(3, _translate("MainWindow", "Right channel"))
        self.TB_spec_file1.setText(_translate("MainWindow", "..."))
        self.TB_spec_file2.setText(_translate("MainWindow", "..."))
        self.BT_spec_open1.setText(_translate("MainWindow", "open"))
        self.BT_spec_open2.setText(_translate("MainWindow", "open"))
        self.BT_spec_loadFiles.setText(_translate("MainWindow", "load files"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabSpectrums), _translate("MainWindow", "Spectrums"))
        self.TB_conv_files.setText(_translate("MainWindow", "..."))
        self.GB_conv_noise.setTitle(_translate("MainWindow", "?????????????????? ?????????????? ?? ????????????"))
        self.LB_conv_mulv.setText(_translate("MainWindow", "?????????????????? ??????????????"))
        self.CB_conv_mulv.setItemText(0, _translate("MainWindow", "1"))
        self.CB_conv_mulv.setItemText(1, _translate("MainWindow", "2"))
        self.CB_conv_mulv.setItemText(2, _translate("MainWindow", "4"))
        self.CB_conv_mulv.setItemText(3, _translate("MainWindow", "8"))
        self.LB_conv_noise1.setText(_translate("MainWindow", "?????????????? ???????? ???1 (0-255)"))
        self.LB_conv_noise2.setText(_translate("MainWindow", "?????????????? ???????? ???2 (0-255)"))
        self.LB_conv_noise3.setText(_translate("MainWindow", "?????????????? ???????? ???3 (0-255)"))
        self.LB_conv_noise4.setText(_translate("MainWindow", "?????????????? ???????? ???4 (0-255)"))
        self.LB_conv_noise5.setText(_translate("MainWindow", "?????????????? ???????? ???5 (0-255)"))
        self.LB_conv_noise6.setText(_translate("MainWindow", "?????????????? ???????? ???6 (0-255)"))
        self.LB_conv_noise7.setText(_translate("MainWindow", "?????????????? ???????? ???7 (0-255)"))
        self.TB_conv_folder.setText(_translate("MainWindow", "..."))
        self.BT_conv_load.setText(_translate("MainWindow", "Load"))
        self.BT_conv_convert.setText(_translate("MainWindow", "Convert"))
        self.BT_conv_save.setText(_translate("MainWindow", "Save"))
        self.CB_conv_steps.setItemText(0, _translate("MainWindow", "1"))
        self.CB_conv_steps.setItemText(1, _translate("MainWindow", "2"))
        self.CB_conv_steps.setItemText(2, _translate("MainWindow", "4"))
        self.CB_conv_steps.setItemText(3, _translate("MainWindow", "8"))
        self.CB_conv_steps.setItemText(4, _translate("MainWindow", "16"))
        self.CB_conv_steps.setItemText(5, _translate("MainWindow", "32"))
        self.CB_conv_steps.setItemText(6, _translate("MainWindow", "64"))
        self.CB_conv_steps.setItemText(7, _translate("MainWindow", "128"))
        self.CB_conv_steps.setItemText(8, _translate("MainWindow", "256"))
        self.CB_conv_steps.setItemText(9, _translate("MainWindow", "512"))
        self.CB_conv_steps.setItemText(10, _translate("MainWindow", "1024"))
        self.CB_conv_steps.setItemText(11, _translate("MainWindow", "2048"))
        self.LB_conv_steps.setText(_translate("MainWindow", "x1"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabConvertSpec), _translate("MainWindow", "ConvertSpectrums"))

