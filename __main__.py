"""
:author: Mardanov Rinat Ildarovich
:email: gyxepm@gmail.com
"""
import sys, json, os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from gui_main import Ui_MainWindow
import sound
import numpy as np
import math

'''
GLOSSARY 44100

BUTTON CONNECTS
self.BT_OBJECT.clicked.connect(self.bt_FUNCTION)

SPINBOX CONNECTS
self.spin_OBJECT.valueChanged.connect(self.spin_FUNCTION)

COMBOBOX CONNECTS
self.CB_OBJECT.currentIndexChanged.connect(self.cb_FUNCTION)

EDIT CONNECTS
self.edit_OBJECT.textChanged.connect(self.edit_FUNCTION)

SIGNALS ON ACTIONS
self.actionOBJECT.triggered.connect(self.actFUNCTION)

BT - Push Button
CB - Combo Box
LE - Line Edit
TE - Text Edit
PTE - Plain Text Edit
SB - Spin Box
LB - Label Box
'''


# TODO: convert menu
# TODO: neural network module


def openFile():
    dialog = QFileDialog()
    dialog.setAcceptMode(QFileDialog.AcceptOpen)
    dialog.setViewMode(QFileDialog.Detail)
    dialog.setFileMode(QFileDialog.ExistingFiles)
    dialog.exec()
    filenames = dialog.selectedFiles()
    if len(filenames) > 0:
        return filenames[0]
    else:
        return ''


def openFiles():
    dialog = QFileDialog()
    dialog.setAcceptMode(QFileDialog.AcceptOpen)
    dialog.setViewMode(QFileDialog.Detail)
    dialog.setFileMode(QFileDialog.ExistingFiles)
    dialog.exec()
    filenames = dialog.selectedFiles()
    return filenames


def saveFiles():
    dialog = QFileDialog()
    dialog.setAcceptMode(QFileDialog.AcceptSave)
    dialog.setViewMode(QFileDialog.Detail)
    dialog.setFileMode(QFileDialog.ExistingFiles)
    dialog.exec()
    filenames = dialog.selectedFiles()
    return filenames


def openDirectory():
    dialog = QFileDialog()
    # dialog.setAcceptMode(QFileDialog.AcceptOpen)
    dialog.setOption(QFileDialog.ShowDirsOnly)
    dialog.setViewMode(QFileDialog.Detail)
    # dialog.exec()
    # print(dialog.getExistingDirectory(), dialog.directory())
    directory = dialog.getExistingDirectory()
    return directory


def split2step(a, step, chunk=2048, ch=2):
    L = len(a)
    ALL = np.transpose(a, axes=(2,0,1)).reshape(ch, L*chunk)
    b = np.empty((0, L-1, chunk, ch), dtype=a.dtype)
    for k in range(chunk//step):
        frame = ALL[:, step*k:(chunk*(L-1)+step*k)]
        frame = np.transpose(
            frame.reshape(ch, L-1, chunk), axes=(1, 2, 0)
        )
        frame = frame.reshape((1,)+frame.shape)
        b = np.concatenate((b, frame), axis=0)
    '''b = np.array(
        [
            np.take(
                a.reshape((a.shape[0] * a.shape[1], ch)),
                [j + step for j in range(0, (a.shape[0] - 1) * a.shape[1])]
                , axis=0
            ).reshape((a.shape[0] - 1, chunk, ch))
            for i in range(0, chunk//step)
        ], dtype=a.dtype)'''
    return b


class Main(Ui_MainWindow):
    def __init__(self, form):
        super().__init__()
        self.setupUi(form)
        form.setWindowTitle('Преобразование голоса © Марданов Ринат')

        self.flags = {
            'BT_spec_open1': [self.BT_spec_open1, True],
            'BT_spec_open2': [self.BT_spec_open2, True],
            'TB_spec_file1': [self.TB_spec_file1, True],
            'TB_spec_file2': [self.TB_spec_file2, True],
            'BT_spec_play': [self.BT_spec_play, False],
            'BT_spec_stop': [self.BT_spec_stop, False],
            'BT_spec_loadFiles': [self.BT_spec_loadFiles, False],
            'CB_spec_sound1': [self.CB_spec_sound1, False],
            'CB_spec_sound2': [self.CB_spec_sound2, False],
            'BT_conv_convert': [self.BT_conv_convert, False],
            'BT_conv_load': [self.BT_conv_load, True],
            'BT_conv_save': [self.BT_conv_save, False],
            'TB_conv_files': [self.TB_conv_files, True],
            'TB_conv_folder': [self.TB_conv_folder, False],
            'CB_conv_mulv': [self.CB_conv_mulv, True],
            'LE_conv_foldersave': [self.LE_conv_foldersave, False],
            'progressBar_conv': [self.progressBar_conv, False],
        }
        self.flags_default = self.flags.copy()

        # Oombo Box connects
        # spec
        self.CB_spec_sound1.currentIndexChanged.connect(self.cb_spec_sound1_change)
        self.CB_spec_sound2.currentIndexChanged.connect(self.cb_spec_sound2_change)
        # conv
        self.CB_conv_mulv.currentIndexChanged.connect(self.cb_conv_mulv_change)
        self.CB_conv_steps.currentIndexChanged.connect(self.cb_conv_steps_change)

        # Button clicks
        # spec
        self.BT_spec_open1.clicked.connect(self.bt_spec_open1_clicked)
        self.BT_spec_open2.clicked.connect(self.bt_spec_open2_clicked)
        self.BT_spec_play.clicked.connect(self.bt_spec_play_clicked)
        self.BT_spec_stop.clicked.connect(self.bt_spec_stop_clicked)
        self.BT_spec_loadFiles.clicked.connect(self.bt_spec_loadfiles_clicked)
        self.TB_spec_file1.clicked.connect(self.tb_spec_file1_clicked)
        self.TB_spec_file2.clicked.connect(self.tb_spec_file2_clicked)
        # conv
        self.BT_conv_convert.clicked.connect(self.bt_conv_convert_clicked)
        self.BT_conv_load.clicked.connect(self.bt_conv_load_clicked)
        self.BT_conv_save.clicked.connect(self.bt_conv_save_clicked)
        self.TB_conv_files.clicked.connect(self.tb_conv_files_clicked)
        self.TB_conv_folder.clicked.connect(self.tb_conv_folder_clicked)
        #

        # Functional
        self.sound = sound.Audio()
        self.graphics = Graphs(self, self.sound)
        self.graphics.animation()

        self.fileConv_paths = []
        self.fileConv_count = 0

    def updateEnables(self):
        if self.flags['CB_spec_sound2'][1] and self.flags['CB_spec_sound1'][1]:
            self.flags['BT_spec_loadFiles'][1] = True
        for k, v in self.flags.items():
            if v[0].isEnabled() != v[1]:
                v[0].setEnabled(v[1])

    # -------------/COMBO BOX FUNCTIONS/-------------
    def cb_conv_mulv_change(self):
        ind = self.CB_conv_mulv.currentIndex()
        l = [self.SB_conv_noise1, self.SB_conv_noise2, self.SB_conv_noise3,
             self.SB_conv_noise4, self.SB_conv_noise5, self.SB_conv_noise6,
             self.SB_conv_noise7]
        for o in l:
            o.setEnabled(False)
        for o in l[:2 ** ind - 1]:
            o.setEnabled(True)
        #print()
        # v = self.CB_conv_mulv.itemText(ind)
        # print(ind**2-1, int(v)+1, type(v))
        self.updateEnables()

    def cb_conv_steps_change(self):
        ind = self.CB_conv_steps.currentIndex()
        self.LB_conv_steps.setText('x'+str(self.sound.CHUNK//2**ind))

        pass

    def cb_spec_sound1_change(self):
        curInd = self.CB_spec_sound1.currentIndex()
        curInd2 = self.CB_spec_sound2.currentIndex()
        if curInd == 1 and curInd2 > 0:
            self.CB_spec_sound2.setCurrentIndex(0)
        elif curInd == 2 and (curInd2 == 1 or curInd2 == 2):
            self.CB_spec_sound2.setCurrentIndex(3)
        elif curInd == 3 and (curInd2 == 1 or curInd2 == 3):
            self.CB_spec_sound2.setCurrentIndex(2)
        self.updateEnables()
        pass

    def cb_spec_sound2_change(self):
        curInd = self.CB_spec_sound2.currentIndex()
        curInd1 = self.CB_spec_sound1.currentIndex()
        if curInd == 1 and curInd1 > 0:
            self.CB_spec_sound1.setCurrentIndex(0)
        elif curInd == 2 and (curInd1 == 1 or curInd1 == 2):
            self.CB_spec_sound1.setCurrentIndex(3)
        elif curInd == 3 and (curInd1 == 1 or curInd1 == 3):
            self.CB_spec_sound1.setCurrentIndex(2)
        self.updateEnables()
        pass

    # -------------\COMBO BOX FUNCTIONS\-------------

    # -------------/BUTTON FUNCTIONS/-------------
    # -------------SPEC-------------
    def bt_spec_open1_clicked(self):
        path = self.LE_spec_file1.text()
        self.sound.wf_datas = []
        # print(path)
        if os.path.exists(path):
            print('Start opening "%s"' % (path,))
            self.sound.readFromFile(path)
            print('opened: "%s"' % (path,))
            print(len(self.sound.wf_datas),
                  self.sound.wf_datas[-1].shape if len(self.sound.wf_datas) > 0 else '')
            # print('Start playing')
            # self.sound.play_data = self.sound.wf_datas[-1]  # - self.sound.AUD // 2
            self.CB_spec_sound1.setEnabled(True)
            pass
        self.flags['CB_spec_sound1'][1] = True
        self.updateEnables()
        pass

    def bt_spec_open2_clicked(self):
        path = self.LE_spec_file2.text()
        # print(path)
        if os.path.exists(path):
            print('Start opening "%s"' % (path,))
            self.sound.readFromFile(path)
            print('opened: "%s"' % (path,))
            print(len(self.sound.wf_datas),
                  self.sound.wf_datas[-1].shape if len(self.sound.wf_datas) > 0 else '')
            # print('Start playing')
            # self.sound.play_data = self.sound.wf_datas[-1]  # - self.sound.AUD // 2
            self.CB_spec_sound2.setEnabled(True)
            pass
        self.flags['CB_spec_sound2'][1] = True
        self.updateEnables()
        pass

    def bt_spec_play_clicked(self):
        # self.sound.play_data = np.array([], dtype=np.int16)
        self.sound.play_ind = 0
        self.updateEnables()
        pass

    def bt_spec_stop_clicked(self):
        # self.sound.play_data = np.array([], dtype=np.int16)
        self.sound.play_ind = len(self.sound.play_data)
        self.updateEnables()
        pass

    def bt_spec_loadfiles_clicked(self):
        c1 = self.CB_spec_sound1.currentIndex()
        c2 = self.CB_spec_sound2.currentIndex()
        f1 = self.sound.wf_datas[0]
        f2 = self.sound.wf_datas[1]
        self.sound.spec_datas = []
        if c1 == 1:
            # both from 1 file
            self.sound.play_data = f1.copy()
            self.sound.spec_datas.append(self.sound.wavToSpec(f1, 0))
            self.sound.spec_datas.append(self.sound.wavToSpec(f1, 1))
            pass
        elif c2 == 1:
            # both from 2 file
            self.sound.play_data = f2.copy()
            self.sound.spec_datas.append(self.sound.wavToSpec(f2, 0))
            self.sound.spec_datas.append(self.sound.wavToSpec(f2, 1))
            pass
        elif c1 == 2 and c2 == 0:
            # left from 1 file
            res = np.transpose(
                np.concatenate((
                    np.transpose(f1, axes=(0, 2, 1))[:, 0, :].reshape((len(f1), 1, self.sound.CHUNK)),
                    np.array([[0] * self.sound.CHUNK] * len(f1), dtype=np.int16).reshape((len(f1), 1, self.sound.CHUNK))
                ), axis=1), axes=(0, 2, 1)
            )
            self.sound.play_data = res.copy()
            self.sound.spec_datas.append(self.sound.wavToSpec(res, 0))
            self.sound.spec_datas.append(self.sound.wavToSpec(res, 1))
            pass
        elif c1 == 3 and c2 == 0:
            # right from 1 file
            res = np.transpose(
                np.concatenate((
                    np.transpose(f1, axes=(0, 2, 1))[:, 1, :].reshape((len(f1), 1, self.sound.CHUNK)),
                    np.array([[0] * self.sound.CHUNK] * len(f1), dtype=np.int16).reshape((len(f1), 1, self.sound.CHUNK))
                ), axis=1), axes=(0, 2, 1)
            )
            self.sound.play_data = res.copy()
            self.sound.spec_datas.append(self.sound.wavToSpec(res, 0))
            self.sound.spec_datas.append(self.sound.wavToSpec(res, 1))
            pass
        elif c1 == 2 and c2 == 3:
            # left from 1 file AND right from 2 file
            # a = np.transpose(f1, axes=(0, 2, 1))[:, 0, :]
            # b = np.transpose(f2, axes=(0, 2, 1))[:, 1, :]
            # print(a.shape, b.shape)
            # print(a[:4], b[:4])
            res = np.transpose(
                np.concatenate((
                    np.transpose(f1, axes=(0, 2, 1))[:, 0, :].reshape((len(f1), 1, self.sound.CHUNK)),
                    np.transpose(f2, axes=(0, 2, 1))[:, 1, :].reshape((len(f2), 1, self.sound.CHUNK))
                ), axis=1), axes=(0, 2, 1)
            )
            self.sound.play_data = res.copy()
            self.sound.spec_datas.append(self.sound.wavToSpec(res, 0))
            self.sound.spec_datas.append(self.sound.wavToSpec(res, 1))
            pass
        elif c1 == 0 and c2 == 2:
            # left from 2 file
            res = np.transpose(
                np.concatenate((
                    np.array([[0] * self.sound.CHUNK] * len(f2), dtype=np.int16).reshape((len(f2), 1, self.sound.CHUNK)),
                    np.transpose(f2, axes=(0, 2, 1))[:, 0, :].reshape((len(f2), 1, self.sound.CHUNK))
                ), axis=1), axes=(0, 2, 1)
            )
            self.sound.play_data = res.copy()
            self.sound.spec_datas.append(self.sound.wavToSpec(res, 0))
            self.sound.spec_datas.append(self.sound.wavToSpec(res, 1))
            # print(self.sound.play_data.shape, self.sound.play_data[0][:4])
            pass
        elif c1 == 3 and c2 == 2:
            # left fromm 2 file AND right from 1 file
            res = np.transpose(
                np.concatenate((
                    np.transpose(f1, axes=(0, 2, 1))[:, 1, :].reshape((len(f1), 1, self.sound.CHUNK)),
                    np.transpose(f2, axes=(0, 2, 1))[:, 0, :].reshape((len(f2), 1, self.sound.CHUNK))
                ), axis=1), axes=(0, 2, 1)
            )
            self.sound.play_data = res.copy()
            self.sound.spec_datas.append(self.sound.wavToSpec(res, 0))
            self.sound.spec_datas.append(self.sound.wavToSpec(res, 1))
            pass
        elif c1 == 0 and c2 == 3:
            # right from 2 file
            res = np.transpose(
                np.concatenate((
                    np.array([[0] * self.sound.CHUNK] * len(f2), dtype=np.int16).reshape((len(f2), 1, self.sound.CHUNK)),
                    np.transpose(f2, axes=(0, 2, 1))[:, 1, :].reshape((len(f2), 1, self.sound.CHUNK))
                ), axis=1), axes=(0, 2, 1)
            )
            self.sound.play_data = res.copy()
            self.sound.spec_datas.append(self.sound.wavToSpec(res, 0))
            self.sound.spec_datas.append(self.sound.wavToSpec(res, 1))
            pass
        else:
            # both empty
            self.sound.play_data = np.array([], dtype=np.int16)
            pass
        self.sound.play_ind = len(self.sound.play_data)
        if self.flags['CB_spec_sound2'][1] and self.flags['CB_spec_sound1'][1]:
            self.flags['BT_spec_play'][1] = True
            self.flags['BT_spec_stop'][1] = True
        print(self.sound.play_data.shape)
        self.updateEnables()
        print('end loads')
        pass

    def tb_spec_file1_clicked(self):
        fs = openFile()
        # t = TH.Thread(target=self.graphics.update, args=())
        # t.start()
        # self.graphics.update()
        # print(fs, type(fs))
        self.LE_spec_file1.setText(fs)
        self.updateEnables()
        pass

    def tb_spec_file2_clicked(self):
        fs = openFile()
        # print(fs)
        self.LE_spec_file2.setText(fs)
        self.updateEnables()
        pass

    # -------------CONV-------------
    def bt_conv_convert_clicked(self):
        print('start convert')
        self.sound.spec_params = []
        self.sound.spec_bytes = {}
        self.sound.spec_datas = []
        ind = self.CB_conv_mulv.currentIndex()
        noises = [self.SB_conv_noise1, self.SB_conv_noise2, self.SB_conv_noise3,
                  self.SB_conv_noise4, self.SB_conv_noise5, self.SB_conv_noise6,
                  self.SB_conv_noise7]
        levels = [int(n.value()) for n in noises]
        # print(sound.add(ind**2 - 1, 1, 0, 9) + 1)
        # print(levels)
        c = int(self.CB_conv_mulv.itemText(ind)) + 1
        # print(c)
        step = 2**self.CB_conv_steps.currentIndex()
        steps = self.sound.CHUNK // step
        L1 = len(self.sound.wf_datas)
        print('Step: ', step, 'Steps: ', steps)
        for i, wf_data in enumerate(self.sound.wf_datas):
            if step == self.sound.CHUNK:
                L2 = len(wf_data)
                print('spec 0')
                spec = self.sound.wavToSpec(wf_data, 0)
                print('spec 0-1')
                self.sound.spec_datas.append(spec)
                # with open(self.fileConv_paths[i]+'_%d.spec' % (0,), mode='wb') as f:
                #     f.write(spec.tobytes())
                self.sound.spec_bytes[str(self.fileConv_paths[i]+'_%d.spec' % (0,))] = spec.tobytes()
                print('spec 1')
                spec = self.sound.wavToSpec(wf_data, 1)
                self.sound.spec_datas.append(spec)
                # with open(self.fileConv_paths[i]+'_%d.spec' % (1,), mode='wb') as f:
                #     f.write(spec.tobytes())
                self.sound.spec_bytes[str(self.fileConv_paths[i] + '_%d.spec' % (1,))] = spec.tobytes()
                for j in range(0, 2**ind-1):
                    # print('spec', j+2, j, 2**ind-1)
                    # print(len(levels), levels[j])
                    noised = sound.gen_noise(wf_data, levels[j])
                    # print(noised[0])
                    # print(wf_data[0])
                    # print((noised-wf_data)[0])
                    # print((noised-wf_data).max(), (noised-wf_data).min())
                    # print(1, j)
                    spec = self.sound.wavToSpec(noised, 1)
                    # print(2, j)
                    # print((spec-self.sound.spec_datas[-1])[0])
                    self.sound.spec_datas.append(spec)
                    # print(3, j)
                    # with open(self.fileConv_paths[i] + '_%d.spec' % (j+2,), mode='wb') as f:
                    #     f.write(spec.tobytes())
                    self.sound.spec_bytes[str(self.fileConv_paths[i] + '_%d.spec' % (j+2,))] = spec.tobytes()
                    # print(4, j)
                # print(self.sound.spec_params)
                self.sound.spec_params.append(
                    {
                        'name': self.fileConv_paths[i]+'_%d.spec',
                        'count': c
                    }
                )
                # print(j)
                self.progressBar_conv.setValue(100*(i+1)/self.fileConv_count)
                # print(j)
            else:
                L2 = len(wf_data)
                print('L2', L2)
                wf_datas_n = [wf_data]
                for j in range(0, 2**ind - 1):
                    print('Gen noise', j + 1)
                    wf_datas_n.append(sound.gen_noise(wf_data, levels[j]))

                wf_datas_nWs = []
                for j in range(len(wf_datas_n)):
                    print('Step gen', j, 'step', step)
                    print('wf_datas_n[j]', wf_datas_n[j].shape)
                    s2s = split2step(wf_datas_n[j], step, self.sound.CHUNK)
                    print('s2s', s2s.shape)
                    wf_datas_nWs.append(
                        np.array(s2s, dtype=np.int16)
                    )

                specs = []
                for j in range(len(wf_datas_nWs)):
                    print('spec conv', j)
                    if j == 0:
                        # print('wf_datas_nWs[j]', wf_datas_nWs[j].shape)
                        # print('wf_datas_nWs[0][0]', wf_datas_nWs[j][0].shape)
                        spec = [self.sound.wavToSpec(k, 0) for k in wf_datas_nWs[j]]
                        # spec = self.sound.wavToSpec(wf_datas_n[j],0)
                        # print('spc.shape', len(spec), spec[0].shape)
                        spec = np.array(spec, dtype=np.float64).reshape(
                             (len(spec)*len(spec[0]), self.sound.CHUNK))
                        specs.append(spec)
                        # print('wf_datas_nWs[1][0]', wf_datas_nWs[j][0].shape)
                        spec = [self.sound.wavToSpec(k, 1) for k in wf_datas_nWs[j]]
                        # spec = self.sound.wavToSpec(wf_datas_n[j], 1)
                        spec = np.array(spec, dtype=np.float64).reshape(
                            (len(spec) * len(spec[0]), self.sound.CHUNK))
                        specs.append(spec)
                        # print('end j=0')
                    else:
                        spec = [self.sound.wavToSpec(k, 1) for k in wf_datas_nWs[j]]
                        spec = np.array(spec, dtype=np.float64).reshape(
                            (len(spec) * len(spec[0]), self.sound.CHUNK))
                        # spec = self.sound.wavToSpec(wf_datas_n[j], 1)
                        specs.append(spec)
                for j in range(len(specs)):
                    self.sound.spec_datas.append(specs[j])
                    self.sound.spec_bytes[
                        str(self.fileConv_paths[i] + '_%d.spec' % (j,))
                    ] = specs[j].tobytes()
                    print(specs[j].max())
                '''
                wf_data_all = np.transpose(wf_data, axes=(2, 0, 1)).reshape(2, L2*2048)
                # print('wf_data_all.shape', wf_data_all.shape)

                wf_data_res = np.empty((0, 2048, 2), dtype=np.int16)
                for k in range(steps):
                    # long ONE frame
                    wf_data_frame = wf_data_all[:, step*k:(2048*(L2-1)+step*k)]
                    # split to standart model (N, 2048, 2)
                    wf_data_frames = np.transpose(
                        wf_data_frame.reshape(2, L2 - 1, 2048), axes=(1, 2, 0)
                    )
                    wf_data_res = np.concatenate((wf_data_res, wf_data_frames), axis=0)
                    pass
                # print('wf_data_res.shape', wf_data_res.shape)
                spec = self.sound.wavToSpec(wf_data_res, 0)
                # print('spec 0-1')
                self.sound.spec_datas.append(spec)
                # with open(self.fileConv_paths[i]+'_%d.spec' % (0,), mode='wb') as f:
                #     f.write(spec.tobytes())
                self.sound.spec_bytes[str(self.fileConv_paths[i] + '_%d.spec' % (0,))] = spec.tobytes()
                # print('spec 1')
                spec = self.sound.wavToSpec(wf_data_res, 1)
                self.sound.spec_datas.append(spec)
                # with open(self.fileConv_paths[i]+'_%d.spec' % (1,), mode='wb') as f:
                #     f.write(spec.tobytes())
                self.sound.spec_bytes[str(self.fileConv_paths[i] + '_%d.spec' % (1,))] = spec.tobytes()
                for j in range(0, 2 ** ind - 1):
                    # print('spec', j+2, j, 2**ind-1)
                    # print(len(levels), levels[j])
                    noised = sound.gen_noise(wf_data_res, levels[j])
                    # print(noised[0])
                    # print(wf_data[0])
                    # print((noised-wf_data)[0])
                    # print((noised-wf_data).max(), (noised-wf_data).min())
                    # print(1, j)
                    spec = self.sound.wavToSpec(noised, 1)
                    # print(2, j)
                    # print((spec-self.sound.spec_datas[-1])[0])
                    self.sound.spec_datas.append(spec)
                    # print(3, j)
                    # with open(self.fileConv_paths[i] + '_%d.spec' % (j+2,), mode='wb') as f:
                    #     f.write(spec.tobytes())
                    self.sound.spec_bytes[str(self.fileConv_paths[i] + '_%d.spec' % (j + 2,))] = spec.tobytes()
                    # print(4, j)
                # print(self.sound.spec_params)
                '''
                self.sound.spec_params.append(
                    {
                        'name': self.fileConv_paths[i] + '_%d.spec',
                        'count': c
                    }
                )
                # print(j)
                self.progressBar_conv.setValue(100 * (i + 1) / self.fileConv_count)
                # print(j)
        self.flags['BT_conv_save'][1] = True
        self.flags['LE_conv_foldersave'][1] = True
        self.flags['TB_conv_folder'][1] = True
        self.updateEnables()
        pass

    def bt_conv_load_clicked(self):
        self.sound.wf_datas = []
        self.sound.spec_datas = []
        self.fileConv_count = 0
        self.fileConv_paths = []
        files = self.PTE_conv_files.toPlainText().split('\n')
        # print(files)
        for file in files:
            if self.sound.readFromFile(file):
                self.fileConv_count += 1
                self.fileConv_paths.append(file[:-4])
        print(len(self.sound.wf_datas), '/', self.fileConv_count, 'loaded')

        self.flags['BT_conv_convert'][1] = True
        self.flags['progressBar_conv'][1] = True
        self.updateEnables()
        pass

    def bt_conv_save_clicked(self):
        pathDir = self.LE_conv_foldersave.text()
        if os.path.exists(pathDir) and os.path.isdir(pathDir):
            print('Folder: '+pathDir)
        else:
            return None
        js = json.dumps({'data': self.sound.spec_params}, indent=4)
        for v in self.sound.spec_params:
            for c in range(v['count']):
                _t = v['name'].replace('\\', '/').split('/')
                old_path, name = '/'.join(_t[:-1]), _t[-1]
                print(pathDir, str(name) % (c,), sep='/')
                # print(self.sound.spec_bytes.keys())
                # print(old_path + '/' + str(str(name) % (c,)))
                data = self.sound.spec_bytes[old_path + '/' + str(str(name) % (c,))]
                # print(len(data))
                with open(pathDir+'/'+str(str(name) % (c,)), mode='wb') as f:
                    f.write(data)

        with open(pathDir+'/'+'info.json', mode='w') as f:
            f.write(js)
        print('saved')
        self.updateEnables()
        pass

    def tb_conv_files_clicked(self):
        files = openFiles()
        print(files)
        self.PTE_conv_files.setPlainText('\n'.join(files))
        self.updateEnables()
        pass

    def tb_conv_folder_clicked(self):
        d = openDirectory()
        print(d)
        self.LE_conv_foldersave.setText(d)
        self.updateEnables()
        pass
    # -------------\BUTTON FUNCTIONS\-------------


class TimeLine(QObject):
    frameChanged = QtCore.pyqtSignal(int)

    def __init__(self, interval=60, loopCount=1, parent=None):
        super(TimeLine, self).__init__(parent)
        self._startFrame = 0
        self._endFrame = 0
        self._loopCount = loopCount
        self._timer = QTimer(self, timeout=self.on_timeout)
        self._counter = 0
        self._loop_counter = 0
        self.setInterval(interval)

    def on_timeout(self):
        if self._startFrame <= self._counter < self._endFrame:
            self.frameChanged.emit(self._counter)
            self._counter += 1
        else:
            self._counter = 0
            self._loop_counter += 1

        if self._loopCount > 0:
            if self._loop_counter >= self.loopCount():
                self._timer.stop()

    def setLoopCount(self, loopCount):
        self._loopCount = loopCount

    def loopCount(self):
        return self._loopCount

    interval = QtCore.pyqtProperty(int, fget=loopCount, fset=setLoopCount)

    def setInterval(self, interval):
        self._timer.setInterval(interval)

    def interval(self):
        return self._timer.interval()

    interval = QtCore.pyqtProperty(int, fget=interval, fset=setInterval)

    def setFrameRange(self, startFrame, endFrame):
        self._startFrame = startFrame
        self._endFrame = endFrame

    @QtCore.pyqtSlot()
    def start(self):
        self._counter = 0
        self._loop_counter = 0
        self._timer.start()


class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.setMouseMode(self.RectMode)

    ## reimplement right-click to zoom out
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            self.autoRange()

    def mouseDragEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            ev.ignore()
        else:
            pg.ViewBox.mouseDragEvent(self, ev)


class Graphs:
    def __init__(self, main: Main, sound: sound.Audio):
        self.main = main
        self.sound = sound
        self.AUD = AUD = self.sound.AUD

        self.iii = 0

        pg.setConfigOption('background', 0.05)
        pg.setConfigOptions(antialias=True)
        self.traces = dict()
        '''
        # vb = CustomViewBox() viewBox=CustomViewBox(),
        self.wf_plot_1 = pg.PlotWidget(enableMenu=True)  # parent=self.main.plot_spec_W1)

        # self.main.gridLayout.addWidget(self.win, 2,0,1,1)
        # self.main.plot_spec_W1 = self.win
        self.wf_plot_1.setParent(self.main.plot_spec_W1)
        # self.wf_plot_1.setAspectLocked(lock=True, ratio=0.01)
        self.wf_plot_1.setYRange(-(AUD//2), AUD//2)
        self.wf_plot_1.setXRange(0, 2048)
        #self.wf_plot_1.scale(1, 0.05)
        '''
        self.wf_plot_1 = pg.PlotWidget()
        self.main.gridLayout.addWidget(self.wf_plot_1, 0, 0)
        self.wf_plot_1.setYRange(-(AUD // 2), AUD // 2)
        self.wf_plot_1.setXRange(0, self.sound.CHUNK)

        self.wf_plot_2 = pg.PlotWidget()
        self.main.gridLayout.addWidget(self.wf_plot_2, 0, 1)
        self.wf_plot_2.setYRange(-(AUD // 2), AUD // 2)
        self.wf_plot_2.setXRange(0, self.sound.CHUNK)

        self.sp_plot_1 = pg.PlotWidget()
        self.main.gridLayout.addWidget(self.sp_plot_1, 1, 0)
        self.sp_plot_1.setYRange(-1, 2)
        self.sp_plot_1.setXRange(0, self.sound.CHUNK)
        # self.sp_plot_1.setXRange(np.log10(20), np.log10(self.sound.RATE / 2), padding=0.005)

        self.sp_plot_2 = pg.PlotWidget()
        self.main.gridLayout.addWidget(self.sp_plot_2, 1, 1)
        self.sp_plot_2.setYRange(-1, 2)
        self.sp_plot_2.setXRange(0, self.sound.CHUNK)
        # self.sp_plot_2.setXRange(np.log10(20), np.log10(self.sound.RATE / 2), padding=0.005)

        self._plots = [
                          PLOT.plot(
                              [],
                              [],
                              pen=pg.mkPen(color='g', width=1)
                          ) for PLOT in [self.wf_plot_1, self.wf_plot_2]
                      ] + [
                          PLOT.plot(
                              [],
                              [],
                              pen=pg.mkPen(color='r', width=1)
                          ) for PLOT in [self.sp_plot_1, self.sp_plot_2]
                      ]
        self._plots[2].setLogMode(True, False)
        # print()
        # self.sp_plot_1.setXRange(0, 2048)
        self._plots[3].setLogMode(True, False)
        # self.sp_plot_2.setXRange(0, 2048)
        self._timeline = TimeLine(loopCount=0, interval=23)
        self._timeline.setFrameRange(0, 720)
        self._timeline.frameChanged.connect(self.generate_data)
        #
        '''
        wf_xlabels = [(0, '0'), (2048, '2048'), (4096, '4096')]
        wf_xaxis = pg.AxisItem(orientation='bottom')
        wf_xaxis.setTicks([wf_xlabels])

        wf_ylabels = [(0, '-' + str(AUD // 2)), (AUD // 2 - 1, '0'), (AUD - 1, str(AUD // 2))]
        wf_yaxis = pg.AxisItem(orientation='left')
        wf_yaxis.setTicks([wf_ylabels])
        print([math.sin(2*math.pi*110*(t+self.iii)) for t in range(2048)])
        self.waveform_1.plot(list(range(2048)),
                             [math.sin(2 * math.pi * 110 * (t + self.iii)) for t in range(2048)]
                             )
        self.waveform_1.plot.setData()'''
        # for i in range(50):
        #    self.update()
        #    time.sleep(20)
        # self.waveform_1.plot([0, 2048, 4096], [-AUD // 2, AUD, AUD // 2])
        # self.waveform = self.win.addPlot(
        #    title='WAVEFORM_1', row=1, col=1, axisItems={'bottom': wf_xaxis, 'left': wf_yaxis},
        # )

        # self.main.gridLayout.addWidget(self.win, 2, 0)
        # self.win.setParentItem(self.main.gridLayout)
        # self.win.setGeometry(0, 0, 100, 100)

    def update(self):
        print('update', self.iii)
        self.iii += 1
        print('update', self.iii)
        # time.sleep(0.02)
        # self.update()
        pass

    def plot_data(self, data):
        for plt, val in zip(self._plots, data):
            plt.setData(range(len(val)), val)

    def set_plotdata(self, name, data_x, data_y):
        if name in self.traces:
            self.traces[name].clear()
            # self.

    def animation(self):
        self._timeline.start()
        # timer.startTimer(1)

    # @QtCore.pyqtSlot(int)
    def generate_data(self, i):
        ang = np.arange(i, i + self.sound.CHUNK)
        sin_func = np.sin(np.radians(ang) * 440) * self.AUD // 4
        # print(sin_func.shape)
        ind = self.sound.play_ind
        L = len(self.sound.play_data)
        if L > 0 and (0 <= ind < L):
            # print('gen', ind)
            _p = np.transpose(self.sound.play_data, axes=(0, 2, 1))
            # print(_p.shape)
            pdata = [_p[:, 0, :], _p[:, 1, :]]
            # print(len(pdata[0][ind]))
            _s = self.sound.spec_datas
            sdata = [_s[0][ind], _s[1][ind]]
            sdata = [np.abs(sdata[j] / (self.AUD // 2 * self.sound.CHUNK)) for j in range(2)]
            # print(sdata[0])
            self.plot_data([pdata[0][ind], pdata[1][ind]] + sdata)
        # else:
        #     pdata = [[0]*2048]*2

        # cos_func = np.cos(np.radians(ang))
        sin_func = np.sin(np.radians(ang) * 440) * self.AUD // 4
        # tan_func = sin_func / cos_func
        # tan_func[(tan_func < -3) | (tan_func > 3)] = np.NaN


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = Main(window)
    window.show()
    sys.exit(app.instance().exec_())
