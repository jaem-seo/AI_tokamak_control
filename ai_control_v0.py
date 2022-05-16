#!/usr/bin/env python

import os, sys, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import pyqtSignal,Qt
from PyQt5.QtWidgets import QApplication,\
                            QPushButton,\
                            QWidget,\
                            QHBoxLayout,\
                            QVBoxLayout,\
                            QGridLayout,\
                            QLabel,\
                            QLineEdit,\
                            QTabWidget,\
                            QTabBar,\
                            QGroupBox,\
                            QDialog,\
                            QTableWidget,\
                            QTableWidgetItem,\
                            QInputDialog,\
                            QMessageBox,\
                            QComboBox,\
                            QShortcut,\
                            QFileDialog,\
                            QCheckBox,\
                            QRadioButton,\
                            QHeaderView,\
                            QSlider,\
                            QSpinBox,\
                            QDoubleSpinBox
from scipy import interpolate
from common.model_structure import *
from common.wall import *
from common.setting import *

# Setting
base_path = os.path.abspath(os.path.dirname(sys.argv[0]))
background_path        = base_path + '/images/insideKSTAR.jpg'
lstm_model_path        = base_path + '/weights/lstm/'
nn_model_path          = base_path + '/weights/nn/'
bpw_model_path         = base_path + '/weights/bpw/'
k2rz_model_path        = base_path + '/weights/k2rz/'
rl_designer_model_path = base_path + '/weights/rl/nf2022/best_model.zip'
rl_k2x_model_path      = base_path + '/weights/rl/k2x/TD3_test_model_1.zip'
max_models = 10
init_models = 4
max_shape_models = 4
decimals = np.log10(1000)
dpi = 1
plot_length = 165
t_delay = 0.002
steady_model = False
bavg = 0.0

# Fixed setting
year_in = 2021
ec_freq = 105.e9

# Matplotlib rcParams setting
rcParamsSetting(dpi)

# RL setting
interval = 20
low_state   = [0.35, 1.68, 0.2, 0.5, 1.265, 2.18, 1.1, 3.8, 0.84, 1.1, 3.8, 0.84, 1.15, 1.15, 0.45]
high_state  = [0.75, 1.9, 0.5, 0.8, 1.34, 2.29, 2.1, 6.2, 1.06, 2.1, 6.2, 1.06, 1.75, 1.75, 0.6]
low_action  = [0.35, 1.68, 0.2, 0.5, 1.265, 2.18]
high_action = [0.75, 1.9, 0.5, 0.8, 1.34, 2.29]
low_state_k2x   = [0.4, 0.5, 1.265, 2.18, 1.6, 0.1, 0.55]
high_state_k2x  = [0.8, 3.0, 1.34, 2.29, 1.9, 0.5, 0.95]
low_action_k2x  = [1.35, 0.7, -0.05]
high_action_k2x = [1.60, 1.05, 0.0]

# Inputs
input_params = ['Ip [MA]','Bt [T]','GW.frac. [-]',\
                'Pnb1a [MW]','Pnb1b [MW]','Pnb1c [MW]',\
                'Pec2 [MW]','Pec3 [MW]','Zec2 [cm]','Zec3 [cm]',\
                'In.Mid. [m]','Out.Mid. [m]','Elon. [-]','Up.Tri. [-]','Lo.Tri. [-]']
input_mins = [0.3,1.5,0.2  , 0.0, 0.0, 0.0, 0.0,0.0,-10,-10, 1.265,2.18,1.6,0.1,0.5 ]
input_maxs = [0.8,2.7,0.6  , 1.75,1.75,1.5, 0.8,0.8, 10, 10, 1.36, 2.29,2.0,0.5,0.9 ]
input_init = [0.5,1.8,0.275, 1.5, 1.5, 0.5, 0.0,0.0,0.0,0.0, 1.32, 2.22,1.7,0.3,0.75]

# Outputs
output_params0 = ['βn','q95','q0','li']
output_params1 = ['βp','wmhd']
output_params2 = ['βn','βp','h89','h98','q95','q0','li','wmhd']
dummy_params = ['Ip [MA]', 'Elon. [-]', 'Up.Tri. [-]', 'Lo.Tri. [-]', 'In.Mid. [m]', 'Out.Mid. [m]']

# Targets
target_params = ['βp','q95','li']
target_mins   = [ 1.1,  3.8,0.84]
target_maxs   = [ 2.1,  6.2,1.06]
target_init   = [ 1.6,  5.0,0.95]

def i2f(i,decimals=decimals):
    return float(i/10**decimals)

def f2i(f,decimals=decimals):
    return int(f*10**decimals)

class KSTARWidget(QDialog):
    def __init__(self, parent=None):
        super(KSTARWidget, self).__init__(parent)

        self.originalPalette = QApplication.palette()
        
        # Initial condition
        self.first = True
        self.time = np.linspace(-0.1*(plot_length-1),0,plot_length)
        self.outputs, self.dummy = {}, {}
        for p in output_params2:
            self.outputs[p] = [0.]
        for p in dummy_params:
            self.dummy[p] = [0.]
        self.x = np.zeros([10,21])
        self.targets = {}
        for i,target_param in enumerate(target_params):
            self.targets[target_param] = [target_init[i],target_init[i]]

        # Load models
        self.k2rz = k2rz(model_path=k2rz_model_path, n_models=max_shape_models)
        if steady_model:
            self.kstar_nn = kstar_nn(model_path=nn_model_path, n_models=max_models)
        else:
            self.kstar_nn = kstar_nn(model_path=nn_model_path, n_models=1)
            self.kstar_lstm = kstar_lstm(model_path=lstm_model_path, n_models=max_models)
        self.bpw_nn = bpw_nn(model_path=bpw_model_path, n_models=max_models)
        
        # Load agents
        self.designer = SB2_model(
            model_path = rl_designer_model_path, 
            low_state = low_state, 
            high_state = high_state, 
            low_action = low_action, 
            high_action = high_action, 
            activation='relu', 
            last_actv='tanh', 
            norm=True, 
            bavg=bavg
        )
        self.rl_k2x = SB2_model(
            model_path = rl_k2x_model_path,
            low_state = low_state_k2x, 
            high_state = high_state_k2x, 
            low_action = low_action_k2x, 
            high_action = high_action_k2x, 
            activation='relu', 
            last_actv='tanh', 
            norm=True, 
            bavg=bavg
        )

        # Top layout
        topLayout = QHBoxLayout()
        
        nModelLabel = QLabel('# of models:')
        self.nModelBox = QSpinBox()
        self.nModelBox.setMinimum(1)
        self.nModelBox.setMaximum(max_models)
        self.nModelBox.setValue(init_models)
        self.resetModelNumber()
        self.nModelBox.valueChanged.connect(self.resetModelNumber)
        
        intervalLabel = QLabel(' Control/Relax interval [0.1s]:')
        self.intervalBox = QSpinBox()
        self.intervalBox.setMinimum(2)
        self.intervalBox.setMaximum(interval)
        self.intervalBox.setValue(interval)
        self.intervalBox.valueChanged.connect(self.resetInterval)
        self.resetInterval()

        self.rtRunPushButton = QPushButton('Run')
        self.rtRunPushButton.setCheckable(True)
        self.rtRunPushButton.setChecked(True)
        self.rtRunPushButton.clicked.connect(self.reCreateOutputBox)

        self.shuffleModelPushButton = QPushButton('Shuffle models')
        self.shuffleModelPushButton.clicked.connect(self.shuffleModels)

        self.plotHeatingCheckBox = QCheckBox('Plot NBI/EC')
        self.plotHeatingCheckBox.setChecked(True)
        self.plotHeatingCheckBox.stateChanged.connect(self.rePlotOutputBox)

        self.plotHeatLoadCheckBox = QCheckBox('Plot heat load')
        self.plotHeatLoadCheckBox.setChecked(True)
        self.plotHeatLoadCheckBox.stateChanged.connect(self.rePlotOutputBox)

        self.overplotCheckBox = QCheckBox('Overlap device')
        self.overplotCheckBox.setChecked(True)
        self.overplotCheckBox.stateChanged.connect(self.rePlotOutputBox)

        topLayout.addWidget(nModelLabel)
        topLayout.addWidget(self.nModelBox)
        topLayout.addWidget(intervalLabel)
        topLayout.addWidget(self.intervalBox)
        #topLayout.addWidget(self.rtRunPushButton)
        topLayout.addWidget(self.shuffleModelPushButton)
        topLayout.addWidget(self.plotHeatingCheckBox)
        topLayout.addWidget(self.plotHeatLoadCheckBox)
        topLayout.addWidget(self.overplotCheckBox)

        # Middle layout
        self.createInputBox()
        self.createOutputBox()
        self.createAutonomousBox()

        # Bottom layout
        self.run1sButton = QPushButton('▶▶ 1s ▶▶')
        self.run1sButton.setFixedWidth(320)
        self.run1sButton.clicked.connect(self.relaxRun1s)
        self.run2sButton = QPushButton('▶▶ 2s ▶▶')
        self.run2sButton.setFixedWidth(320)
        self.run2sButton.clicked.connect(self.relaxRun2s)
        self.dumpButton = QPushButton('Dump outputs')
        self.dumpButton.setFixedWidth(800)
        self.dumpButton.clicked.connect(self.dumpOutput)
        self.autoButton = QPushButton('AI control')
        self.autoButton.setFixedWidth(120)
        self.autoButton.clicked.connect(self.autoControl)

        # Main layout
        self.mainLayout = QGridLayout()

        self.mainLayout.addLayout(topLayout,0,0,1,2)
        #self.mainLayout.addWidget(self.inputBox,1,0)
        self.mainLayout.addWidget(self.outputBox,1,0,1,2)
        self.mainLayout.addWidget(self.autonomousBox,1,2)
        #self.mainLayout.addWidget(self.run1sButton,2,0)
        #self.mainLayout.addWidget(self.run2sButton,2,1)
        self.mainLayout.addWidget(self.dumpButton,2,0)
        self.mainLayout.addWidget(self.autoButton,2,2)
        
        self.setLayout(self.mainLayout)

        self.setWindowTitle("AI-controlled KSTAR tokamak v0")
        self.tmp = 0

    def resetModelNumber(self):
        if steady_model:
            self.kstar_nn.nmodels = self.nModelBox.value()
        else:
            self.kstar_lstm.nmodels = self.nModelBox.value()
        self.bpw_nn.nmodels = self.nModelBox.value()
        #self.k2rz.nmodels = self.nModelBox.value()

    def resetInterval(self):
        self.interval = self.intervalBox.value()

    def createInputBox(self):
        self.inputBox = QGroupBox('Input parameters')
        layout = QGridLayout()

        self.inputSliderDict = {}
        self.inputValueLabelDict = {}
        for input_param in input_params:
            idx = input_params.index(input_param)
            inputLabel = QLabel(input_param)
            self.inputSliderDict[input_param] = QSlider(Qt.Horizontal, self.inputBox)
            self.inputSliderDict[input_param].setMinimum(f2i(input_mins[idx]))
            self.inputSliderDict[input_param].setMaximum(f2i(input_maxs[idx]))
            self.inputSliderDict[input_param].setValue(f2i(input_init[idx]))
            self.inputSliderDict[input_param].valueChanged.connect(self.updateInputs)
            self.inputValueLabelDict[input_param] = QLabel(f'{self.inputSliderDict[input_param].value()/10**decimals:.3f}')
            self.inputValueLabelDict[input_param].setMinimumWidth(40)

            layout.addWidget(inputLabel,idx,0)
            layout.addWidget(self.inputSliderDict[input_param],idx,1)
            layout.addWidget(self.inputValueLabelDict[input_param],idx,2)

            #for widget in inputLabel,self.inputSliderDict[input_param],self.inputValueLabelDict[input_param]:
            #    widget.setMaximumWidth(30)
        self.runSlider = QSlider(Qt.Horizontal, self.inputBox)
        self.runSlider.setMinimum(0)
        self.runSlider.setMaximum(100)
        self.runSlider.setValue(0)
        self.runSlider.valueChanged.connect(self.updateInputs)
        self.runLabel = QLabel('0.1s ▶')

        layout.addWidget(QLabel('Run only'),len(input_params),0)
        layout.addWidget(self.runSlider,len(input_params),1)
        layout.addWidget(self.runLabel,len(input_params),2)

        self.inputBox.setLayout(layout)
        self.inputBox.setMaximumWidth(320)

    def updateInputs(self):
        for input_param in input_params:
            self.inputValueLabelDict[input_param].setText(f'{self.inputSliderDict[input_param].value()/10**decimals:.3f}')
        if self.rtRunPushButton.isChecked() and time.time()-self.tmp>t_delay:
            self.reCreateOutputBox()
            self.tmp = time.time()

    def updateInputs_without_plot(self):
        if self.rtRunPushButton.isChecked() and time.time()-self.tmp>t_delay:
            self.predict0d(steady=steady_model)

    def createOutputBox(self):
        self.outputBox = QGroupBox('AI control output')

        self.fig = plt.figure(figsize=(6*(100/dpi),4*(100/dpi)),dpi=dpi)
        self.plotPlasma()
        self.canvas = FigureCanvas(self.fig)

        self.layout = QGridLayout()
        self.layout.addWidget(self.canvas)

        self.outputBox.setLayout(self.layout)

    def reCreateOutputBox(self,predict=True):
        self.outputBox = QGroupBox(' ')

        plt.clf()
        self.plotPlasma(predict=predict)
        self.canvas = FigureCanvas(self.fig)

        self.layout = QGridLayout()
        self.layout.addWidget(self.canvas)

        self.outputBox.setLayout(self.layout)
        #self.mainLayout.replaceWidget(self.mainLayout.itemAtPosition(1,1).widget(),self.outputBox)
        self.mainLayout.addWidget(self.outputBox,1,0,1,2)

    def rePlotOutputBox(self):
        self.reCreateOutputBox(predict=False)

    def createAutonomousBox(self):
        self.autonomousBox = QGroupBox('Target setting')

        layout = QGridLayout()

        self.targetSliderDict = {}
        self.targetValueLabelDict = {}

        for target_param in target_params:
            idx = target_params.index(target_param)
            targetLabel = QLabel(target_param)
            targetLabel.setAlignment(Qt.AlignCenter)
            targetLabel.setMaximumWidth(40)
            self.targetSliderDict[target_param] = QSlider(Qt.Vertical, self.autonomousBox)
            self.targetSliderDict[target_param].setMinimum(f2i(target_mins[idx]))
            self.targetSliderDict[target_param].setMaximum(f2i(target_maxs[idx]))
            self.targetSliderDict[target_param].setValue(f2i(target_init[idx]))
            self.targetSliderDict[target_param].valueChanged.connect(self.updateTargets)
            self.targetValueLabelDict[target_param] = QLabel(f'{self.targetSliderDict[target_param].value()/10**decimals:.3f}')
            self.targetValueLabelDict[target_param].setMinimumWidth(40)

            layout.addWidget(targetLabel,idx,0)
            layout.addWidget(self.targetSliderDict[target_param],idx,1)
            layout.addWidget(self.targetValueLabelDict[target_param],idx,2)
        
        self.autonomousBox.setLayout(layout)
        self.autonomousBox.setMaximumWidth(120)

    def updateTargets(self):
        for target_param in target_params:
            self.targetValueLabelDict[target_param].setText(f'{self.targetSliderDict[target_param].value()/10**decimals:.3f}')

    def autoControl(self):
        idx_convert = [0, 12, 13, 14, 10, 11]
        observation = np.zeros_like(low_state)
        observation[:6] = [i2f(self.inputSliderDict[input_params[i]].value()) for i in idx_convert]
        observation[6:9] = [self.outputs[output_params2[i]][-1] for i in [1, 4, 6]]
        observation[9:12] = [i2f(self.targetSliderDict[target_params[i]].value()) for i in [0, 1, 2]]
        observation[12:] = [i2f(self.inputSliderDict[input_params[i]].value()) for i in [3, 4, 5]]
        new_action = self.designer.predict(observation)

        current_action = [i2f(self.inputSliderDict[input_params[i]].value()) for i in idx_convert]
        daction = (new_action - current_action) / self.interval
        for i in range(self.interval): # Control phase
            current_action += daction
            for i, idx in enumerate(idx_convert):
                self.inputSliderDict[input_params[idx]].setValue(f2i(current_action[i]))
            time.sleep(t_delay)
        for i in range(self.interval): # Relaxation phase
            #self.reCreateOutputBox()
            #time.sleep(t_delay)
            if i < self.interval - 1:
                self.predict0d(steady=False)
            else:
                self.reCreateOutputBox()

    def plotPlasma(self,predict=True):
        # Predict plasma
        if predict:
            self.predictBoundary()
            if self.first or steady_model:
                self.predict0d(steady=True)
            else:
                self.predict0d(steady=False)
        ts = self.time[-len(self.outputs['βn']):]
        
        # Plot 2D view
        plt.subplot(1,3,1)
        plt.title('AI-designed plasma shape')
        if self.overplotCheckBox.isChecked():
            self.plotBackground()
            plt.fill_between(self.rbdry,self.zbdry,color='b',alpha=0.2,linewidth=0.0)
        plt.plot(Rwalls,Zwalls,'k',linewidth=1.5*(100/dpi),label='Wall')
        plt.plot(self.rbdry,self.zbdry,'b',linewidth=2*(100/dpi),label='LCFS')
        if self.plotHeatingCheckBox.isChecked():
            self.plotHeating()
        if self.plotHeatLoadCheckBox.isChecked():
            self.plotHeatLoads()
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        if self.overplotCheckBox.isChecked():
            self.plotXpoints()
            plt.xlim([1.1,2.4])
            plt.ylim([-1.6,1.6])
        else:
            plt.axis('scaled')
            plt.grid(linewidth=0.5*(100/dpi))
            plt.legend(loc='center',fontsize=7.5*(100/dpi),markerscale=0.7,frameon=False)
        
        # Plot operation trajectory
        plt.subplot(3,3,2)
        plt.title('AI operation trajectory')
        plt.plot(ts,self.dummy['Ip [MA]'],'k',linewidth=2*(100/dpi),label='Ip [MA]')
        plt.grid(linewidth=0.5*(100/dpi))
        plt.legend(loc='upper left',fontsize=7.5*(100/dpi),frameon=False)
        plt.xlim([-0.1*plot_length-0.2,0.2])
        plt.ylim([0.3, 0.8])
        plt.xticks(color='w')

        plt.subplot(3,3,5)
        plt.plot(ts,np.array(self.dummy['Elon. [-]']) - 1,'k',linewidth=2*(100/dpi),label='Elon.-1')
        plt.plot(ts,self.dummy['Up.Tri. [-]'],'lightgrey',linewidth=2*(100/dpi),label='Up.Tri.')
        plt.plot(ts,self.dummy['Lo.Tri. [-]'],'grey',linewidth=2*(100/dpi),label='Lo.Tri.')
        plt.grid(linewidth=0.5*(100/dpi))
        plt.legend(loc='upper left',fontsize=7.5*(100/dpi),frameon=False)
        plt.xlim([-0.1*plot_length-0.2,0.2])
        plt.ylim([0.15, 1])
        plt.xticks(color='w')

        plt.subplot(3,3,8)
        plt.plot(ts,np.array(self.dummy['In.Mid. [m]']) - 1.265,'k',linewidth=2*(100/dpi),label='Ingap [m]')
        plt.plot(ts,2.316 - np.array(self.dummy['Out.Mid. [m]']),'grey',linewidth=2*(100/dpi),label='Outgap [m]')
        plt.grid(linewidth=0.5*(100/dpi))
        plt.legend(loc='upper left',fontsize=7.5*(100/dpi),frameon=False)
        plt.xlim([-0.1*plot_length-0.2,0.2])
        plt.ylim([0, 0.14])
        plt.xlabel('Relative time [s]')

        # Plot 0D evolution
        alpha = 0.5
        period = 2 * 0.1 * self.interval
        is_relax = ts % period > 0.5 * period
        
        plt.subplot(3,3,3)
        plt.title('Response and target')
        plt.plot(ts,self.outputs['βp'],'k',linewidth=2*(100/dpi),label='βp')
        #plt.plot(ts,self.targets['βp'],'k',linestyle='--',linewidth=2*(100/dpi),label='Target')
        for i in range(int((max(ts) - min(ts)) / period) + 1):
            label = 'Target' if i == 0 else None
            plt.plot(ts[::-1][2 * i * self.interval + 1 : (2 * i + 1) * self.interval], self.targets['βp'][::-1][2 * i * self.interval + 1: (2 * i + 1) * self.interval], 'b', alpha=alpha, linewidth=4*(100/dpi), label=label)
        plt.grid(linewidth=0.5*(100/dpi))
        plt.legend(loc='upper left',fontsize=7.5*(100/dpi),frameon=False)
        plt.xlim([-0.1*plot_length-0.2,0.2])
        gap = 0.1*(target_maxs[0] - target_mins[0])
        plt.ylim([target_mins[0] - gap, target_maxs[0] + gap])
        plt.xticks(color='w')

        plt.subplot(3,3,6)
        plt.plot(ts,self.outputs['q95'],'k',linewidth=2*(100/dpi),label='q95')
        #plt.plot(ts,self.targets['q95'],'k',linestyle='--',linewidth=2*(100/dpi),label='Target')
        for i in range(int((max(ts) - min(ts)) / period) + 1):
            label = 'Target' if i == 0 else None
            plt.plot(ts[::-1][2 * i * self.interval + 1: (2 * i + 1) * self.interval], self.targets['q95'][::-1][2 * i * self.interval + 1: (2 * i + 1) * self.interval], 'b', alpha=alpha, linewidth=4*(100/dpi), label=label)
        plt.grid(linewidth=0.5*(100/dpi))
        plt.legend(loc='upper left',fontsize=7.5*(100/dpi),frameon=False)
        plt.xlim([-0.1*plot_length-0.2,0.2])
        gap = 0.1*(target_maxs[1] - target_mins[1])
        plt.ylim([target_mins[1] - gap, target_maxs[1] + gap])
        plt.xticks(color='w')

        plt.subplot(3,3,9)
        plt.plot(ts,self.outputs['li'],'k',linewidth=2*(100/dpi),label='li')
        #plt.plot(ts,self.targets['li'],'k',linestyle='--',linewidth=2*(100/dpi),label='Target')
        for i in range(int((max(ts) - min(ts)) / period) + 1):
            label = 'Target' if i == 0 else None
            plt.plot(ts[::-1][2 * i * self.interval + 1: (2 * i + 1) * self.interval], self.targets['li'][::-1][2 * i * self.interval + 1: (2 * i + 1) * self.interval], 'b', alpha=alpha, linewidth=4*(100/dpi), label=label)
        plt.grid(linewidth=0.5*(100/dpi))
        plt.legend(loc='upper left',fontsize=7.5*(100/dpi),frameon=False)
        plt.xlim([-0.1*plot_length-0.2,0.2])
        gap = 0.1*(target_maxs[2] - target_mins[2])
        plt.ylim([target_mins[2] - gap, target_maxs[2] + gap])

        plt.xlabel('Relative time [s]')
        plt.tight_layout(h_pad=0., rect=(0.05,0.05,0.95,0.95))

        self.first = False

    def predictBoundary(self):
        ip = self.inputSliderDict[input_params[0]].value()/10**decimals
        bt = self.inputSliderDict[input_params[1]].value()/10**decimals
        bp = self.outputs['βp'][-1]
        rin = self.inputSliderDict[input_params[10]].value()/10**decimals
        rout = self.inputSliderDict[input_params[11]].value()/10**decimals
        k = self.inputSliderDict[input_params[12]].value()/10**decimals
        du = self.inputSliderDict[input_params[13]].value()/10**decimals
        dl = self.inputSliderDict[input_params[14]].value()/10**decimals

        self.k2rz.set_inputs(ip,bt,bp,rin,rout,k,du,dl)
        self.rbdry,self.zbdry = self.k2rz.predict(post=True)
        self.rx1 = self.rbdry[np.argmin(self.zbdry)]
        self.zx1 = np.min(self.zbdry)
        self.rx2 = self.rx1
        self.zx2 = -self.zx1

    def plotXpoints(self,mode=0,zorder=100):
        if mode==0:
            self.rx1 = self.rbdry[np.argmin(self.zbdry)]
            self.zx1 = np.min(self.zbdry)
            self.rx2 = self.rx1
            self.zx2 = -self.zx1
        plt.scatter([self.rx1,self.rx2],[self.zx1,self.zx2],marker='x',color='w',s=100*(100/dpi)**2,linewidths=2*(100/dpi),label='X-points',zorder=zorder)

    def plotHeatLoads(self,n=10,both_side=True):
        kinds = ['linear','quadratic'] #,'cubic']
        wallPath = Path(np.array([Rwalls,Zwalls]).T)
        idx1 = list(self.zbdry).index(self.zx1)
        for kind in kinds:
            f = interpolate.interp1d(self.rbdry[idx1-5:idx1],self.zbdry[idx1-5:idx1],kind=kind,fill_value='extrapolate')
            rsol1 = np.linspace(self.rbdry[idx1],np.min(Rwalls)+1.e-4,n)
            zsol1 = np.array([f(r) for r in rsol1])
            is_inside1 = wallPath.contains_points(np.array([rsol1,zsol1]).T)
            
            f = interpolate.interp1d(self.zbdry[idx1+5:idx1:-1],self.rbdry[idx1+5:idx1:-1],kind=kind,fill_value='extrapolate')
            zsol2 = np.linspace(self.zbdry[idx1],np.min(Zwalls)+1.e-4,n)
            rsol2 = np.array([f(z) for z in zsol2])
            is_inside2 = wallPath.contains_points(np.array([rsol2,zsol2]).T)
            if not np.all(zsol1[is_inside1]>self.zbdry[idx1+1]):
                plt.plot(rsol1[is_inside1],zsol1[is_inside1],'r',linewidth=1.5*(100/dpi))
            plt.plot(rsol2[is_inside2],zsol2[is_inside2],'r',linewidth=1.5*(100/dpi))
            if both_side:
                plt.plot(self.rbdry[idx1-4:idx1+4],-self.zbdry[idx1-4:idx1+4],'b',linewidth=2*(100/dpi),alpha=0.1)
                plt.plot(rsol1[is_inside1],-zsol1[is_inside1],'r',linewidth=1.5*(100/dpi),alpha=0.2)
                plt.plot(rsol2[is_inside2],-zsol2[is_inside2],'r',linewidth=1.5*(100/dpi),alpha=0.2)
        for kind in kinds:
            f = interpolate.interp1d(self.rbdry[idx1-5:idx1+1],self.zbdry[idx1-5:idx1+1],kind=kind,fill_value='extrapolate')
            rsol1 = np.linspace(self.rbdry[idx1],np.min(Rwalls)+1.e-4,n)
            zsol1 = np.array([f(r) for r in rsol1])
            is_inside1 = wallPath.contains_points(np.array([rsol1,zsol1]).T)

            f = interpolate.interp1d(self.zbdry[idx1+5:idx1-1:-1],self.rbdry[idx1+5:idx1-1:-1],kind=kind,fill_value='extrapolate')
            zsol2 = np.linspace(self.zbdry[idx1],np.min(Zwalls)+1.e-4,n)
            rsol2 = np.array([f(z) for z in zsol2])
            is_inside2 = wallPath.contains_points(np.array([rsol2,zsol2]).T)
            if not np.all(zsol1[is_inside1]>self.zbdry[idx1+1]):
                plt.plot(rsol1[is_inside1],zsol1[is_inside1],'r',linewidth=1.5*(100/dpi))
            plt.plot(rsol2[is_inside2],zsol2[is_inside2],'r',linewidth=1.5*(100/dpi))
            if both_side:
                plt.plot(rsol1[is_inside1],-zsol1[is_inside1],'r',linewidth=1.5*(100/dpi),alpha=0.2)
                plt.plot(rsol2[is_inside2],-zsol2[is_inside2],'r',linewidth=1.5*(100/dpi),alpha=0.2)
        plt.plot([self.rx1],[self.zx1],'r',linewidth=1*(100/dpi),label='Heat load')

    def plotBackground(self):
        img = plt.imread(background_path)
        plt.imshow(img,extent=[-1.6,2.45,-1.5,1.35])

    def plotHeating(self):
        pnb1a = self.inputSliderDict['Pnb1a [MW]'].value()/10**decimals
        pnb1b = self.inputSliderDict['Pnb1b [MW]'].value()/10**decimals
        pnb1c = self.inputSliderDict['Pnb1c [MW]'].value()/10**decimals
        pec2 = self.inputSliderDict['Pec2 [MW]'].value()/10**decimals
        pec3 = self.inputSliderDict['Pec3 [MW]'].value()/10**decimals
        zec2 = self.inputSliderDict['Zec2 [cm]'].value()/10**decimals
        zec3 = self.inputSliderDict['Zec3 [cm]'].value()/10**decimals
        bt = self.inputSliderDict['Bt [T]'].value()/10**decimals
        
        rt1,rt2,rt3 = 1.486,1.720,1.245
        w,h = 0.13,0.45
        plt.fill_between([rt1-w/2,rt1+w/2],[-h/2,-h/2],[h/2,h/2],color='g',alpha=0.9 if pnb1a>=0.5 else 0.3)
        plt.fill_between([rt2-w/2,rt2+w/2],[-h/2,-h/2],[h/2,h/2],color='g',alpha=0.9 if pnb1b>=0.5 else 0.3)
        plt.fill_between([rt3-w/2,rt3+w/2],[-h/2,-h/2],[h/2,h/2],color='g',alpha=0.9 if pnb1c>=0.5 else 0.3\
                         ,label='NBI')

        for ns in [1,2,3]:
            rs = 1.60219e-19*1.8*bt/(2.*np.pi*9.10938e-31*ec_freq)*ns
            if min(Rwalls)<rs<max(Rwalls):
                break
        dz = 0.05
        rpos,zpos = 2.449,0.35
        zres = zpos + (zec2/100-zpos)*(rs-rpos)/(1.8-rpos)
        plt.fill_between([rs,rpos],[zres-dz,zpos],[zres+dz,zpos],color='orange',alpha=0.9 if pec2>0.2 else 0.3)
        rpos,zpos = 2.451,-0.35
        zres = zpos + (zec3/100-zpos)*(rs-rpos)/(1.8-rpos)
        plt.fill_between([rs,rpos],[zres-dz,zpos],[zres+dz,zpos],color='orange',alpha=0.9 if pec3>0.2 else 0.3,\
                         label='ECH')

    def predict0d(self,steady=True):
        # Predict output_params0 (βn, q95, q0, li)
        if steady:
            x = np.zeros(17)
            idx_convert = [0,1,3,4,5,6,7,8,9,10,11,12,13,14,10,2]
            for i in range(len(x)-1):
                x[i] = self.inputSliderDict[input_params[idx_convert[i]]].value()/10**decimals
            x[9],x[10] = 0.5*(x[9]+x[10]),0.5*(x[10]-x[9])
            x[14] = 1 if x[14]>1.265+1.e-4 else 0
            x[-1] = year_in
            y = self.kstar_nn.predict(x)
            for i in range(len(output_params0)):
                if len(self.outputs[output_params0[i]]) >= plot_length:
                    del self.outputs[output_params0[i]][0]
                elif len(self.outputs[output_params0[i]]) == 1:
                    self.outputs[output_params0[i]][0] = y[i]
                self.outputs[output_params0[i]].append(y[i])
            self.x[:,:len(output_params0)] = y
            self.x[:,len(output_params0):] = x
        else:
            self.x[:-1,len(output_params0):] = self.x[1:,len(output_params0):]
            idx_convert = [0,1,3,4,5,6,7,8,9,10,11,12,13,14,10,2]
            for i in range(len(self.x[0])-1-4):
                self.x[-1,i+4] = self.inputSliderDict[input_params[idx_convert[i]]].value()/10**decimals
            self.x[-1,13],self.x[-1,14] = 0.5*(self.x[-1,13]+self.x[-1,14]),0.5*(self.x[-1,14]-self.x[-1,13])
            self.x[-1,18] = 1 if self.x[-1,18]>1.265+1.e-4 else 0
            y = self.kstar_lstm.predict(self.x)
            self.x[:-1,:len(output_params0)] = self.x[1:,:len(output_params0)]
            self.x[-1,:len(output_params0)] = y
            for i in range(len(output_params0)):
                if len(self.outputs[output_params0[i]]) >= plot_length:
                    del self.outputs[output_params0[i]][0]
                elif len(self.outputs[output_params0[i]]) == 1:
                    self.outputs[output_params0[i]][0] = y[i]
                self.outputs[output_params0[i]].append(y[i])

        # Update output targets (βp, q95, li)
        if not self.first:
            for i,target_param in enumerate(target_params):
                if len(self.targets[target_param]) >= plot_length:
                    del self.targets[target_param][0]
                elif len(self.targets[target_param]) == 1:
                    self.targets[target_param][0] = i2f(self.targetSliderDict[target_param].value())
                self.targets[target_param].append(i2f(self.targetSliderDict[target_param].value()))

        # Predict output_params1 (βp, wmhd)
        x = np.zeros(8)
        idx_convert = [0,0,1,10,11,12,13,14]
        x[0] = self.outputs['βn'][-1]
        for i in range(1,len(x)):
            x[i] = self.inputSliderDict[input_params[idx_convert[i]]].value()/10**decimals
        x[3],x[4] = 0.5*(x[3]+x[4]),0.5*(x[4]-x[3])
        y = self.bpw_nn.predict(x)
        for i in range(len(output_params1)):
            if len(self.outputs[output_params1[i]]) >= plot_length:
                del self.outputs[output_params1[i]][0]
            elif len(self.outputs[output_params1[i]]) == 1:
                self.outputs[output_params1[i]][0] = y[i]
            self.outputs[output_params1[i]].append(y[i])

        # Store dummy parameters (Ip)
        for p in dummy_params:
            if len(self.dummy[p]) >= plot_length:
                del self.dummy[p][0]
            elif len(self.dummy[p]) == 1:
                self.dummy[p][0] = i2f(self.inputSliderDict[p].value())
            self.dummy[p].append(i2f(self.inputSliderDict[p].value()))

        # Estimate H factors (h89, h98)
        ip = self.inputSliderDict['Ip [MA]'].value()/10**decimals
        bt = self.inputSliderDict['Bt [T]'].value()/10**decimals
        fgw = self.inputSliderDict['GW.frac. [-]'].value()/10**decimals
        ptot = max(self.inputSliderDict['Pnb1a [MW]'].value()/10**decimals \
               + self.inputSliderDict['Pnb1b [MW]'].value()/10**decimals \
               + self.inputSliderDict['Pnb1c [MW]'].value()/10**decimals \
               + self.inputSliderDict['Pec2 [MW]'].value()/10**decimals \
               + self.inputSliderDict['Pec3 [MW]'].value()/10**decimals \
               , 1.e-1) # Not to diverge
        rin = self.inputSliderDict['In.Mid. [m]'].value()/10**decimals
        rout = self.inputSliderDict['Out.Mid. [m]'].value()/10**decimals
        k = self.inputSliderDict['Elon. [-]'].value()/10**decimals

        rgeo,amin = 0.5*(rin+rout),0.5*(rout-rin)
        ne = fgw*10*(ip/(np.pi*amin**2))
        m = 2.0 # Mass number

        tau89 = 0.038*ip**0.85*bt**0.2*ne**0.1*ptot**-0.5*rgeo**1.5*k**0.5*(amin/rgeo)**0.3*m**0.5
        tau98 = 0.0562*ip**0.93*bt**0.15*ne**0.41*ptot**-0.69*rgeo**1.97*k**0.78*(amin/rgeo)**0.58*m**0.19
        h89 = 1.e-6*self.outputs['wmhd'][-1]/ptot/tau89
        h98 = 1.e-6*self.outputs['wmhd'][-1]/ptot/tau98

        if len(self.outputs['h89']) >= plot_length:
            del self.outputs['h89'][0], self.outputs['h98'][0]
        elif len(self.outputs['h89']) == 1:
            self.outputs['h89'][0], self.outputs['h98'][0] = h89, h98

        self.outputs['h89'].append(h89)
        self.outputs['h98'].append(h98)

    def shuffleModels(self):
        np.random.shuffle(self.k2rz.models)
        if steady_model:
            np.random.shuffle(self.kstar_nn.models)
        else:
            np.random.shuffle(self.kstar_lstm.models)
        np.random.shuffle(self.bpw_nn.models)
        print('Models shuffled!')
    
    def relaxRun1s(self):
        for i in range(10 - 1):
            self.predict0d(steady = self.first or steady_model)
        self.reCreateOutputBox()
        self.tmp = time.time()

    def relaxRun2s(self):
        for i in range(20 - 1):
            self.predict0d(steady = self.first or steady_model)
        self.reCreateOutputBox()
        self.tmp = time.time()

    def dumpOutput(self):
        print('\nTrajectories:')
        print(f"Time [s]: {self.time[-len(self.outputs['βn']):]}")
        for dummy in dummy_params:
            print(f'{dummy}: {self.dummy[dummy]}')
        for output in output_params2:
            print(f'{output}: {self.outputs[output]}')
        print('\nCurrent operation control by AI:')
        for input_param in input_params:
            print(f'{input_param}: {i2f(self.inputSliderDict[input_param].value())}')

    def autoRun05(self):
        for i in range(1):
            self.autoAction()
            #time.sleep(t_delay)
            #if self.first or steady_model:
            #    self.predict0d(steady=True)
            #else:
            #    self.predict0d(steady=False)
        #self.reCreateOutputBox()
        self.tmp = time.time()


if __name__ == '__main__':
    app = QApplication([])
    window = KSTARWidget()
    window.show()
    app.exec()

