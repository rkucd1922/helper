import os, sys 
sys.path.append("/global/u2/r/rk7837kl/lib/")
from kul_tools import KulTools as KT

from ase import Atom, Atoms
from ase.io import read, write, Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution as MBD
from ase.optimize import BFGS
from ase import units
from ase.constraints import FixAtoms
from ase.visualize.plot import plot_atoms
from ase.db import connect
from ase.build import molecule,fcc111

import datetime

import numpy as np

from deepmd.calculator import DP
from deepmd.infer import calc_model_devi
from deepmd.infer import DeepPot
from deepmd.infer.model_devi import calc_model_devi_v, calc_model_devi_f

import os

import glob

import shutil 

import scipy.stats as stat
from scipy.interpolate import interpn

from sklearn.metrics import mean_squared_error as MSE

import matplotlib.pyplot as plt

import json

import requests

from PyPDF2 import PdfFileMerger

import seaborn as sns

def makeDPMDvDFTPlots_old(names = None, DPMD_data = None, DFT_file_path = None, DFT_idx = None, model = None, atom_num = None):
    """
    names = names that you want to label the graphs of the parity plots of each system
    DFT_file_path = list containing the file paths to the globbed DFT data
    model = path to the model file
    """
    working_dir = os.getcwd()

    os.mkdir("./DPMDvDFT_graphs")
    
    keys = []
    for i in range(len(names)):  
        keys.append(i)
        
    config_idx = dict(zip(keys,names))
                          
    
    # write a file that houses all of the MSEs
    file = open("./DPMDvDFT_MSEs.txt","a")
    
    for path_idx in range(len(names)):
        DPMDFx = []
        DFTFx = []
        DPMDFy = []
        DFTFy = []
        DPMDFz = []
        DFTFz = []
        DPMDE = []
        DFTE = []
        MD_step = []
        
        # append the DPMD data
        DPMD = read(DPMD_data[int(path_idx)], index = ":")
        for i,atoms in enumerate(DPMD):
            DPMDFx.append(atoms.get_forces()[:,0][atom_num])
            DPMDFy.append(atoms.get_forces()[:,1][atom_num])
            DPMDFz.append(atoms.get_forces()[:,2][atom_num])
            DPMDE.append(atoms.get_potential_energy())
            MD_step.append(i*100)
            
        DFT = read(DFT_file_path[int(path_idx)], index = ":")
        for atoms in DFT:
            DFTFx.append(atoms.get_forces()[:,0][atom_num])
            DFTFy.append(atoms.get_forces()[:,1][atom_num])
            DFTFz.append(atoms.get_forces()[:,2][atom_num])
            DFTE.append(atoms.get_potential_energy())

        DPMD_for_DFT_Fx = []
        DPMD_for_DFT_Fy = []
        DPMD_for_DFT_Fz = []
        DPMD_for_DFT_E = []
        
        for idx in DFT_idx:
            DPMD_for_DFT_Fx.append(DPMDFx[idx])
            DPMD_for_DFT_Fy.append(DPMDFy[idx])
            DPMD_for_DFT_Fz.append(DPMDFz[idx])
            DPMD_for_DFT_E.append(DPMDE[idx])
        
        MSE_Fx = MSE(y_true = DFTFx, y_pred = DPMD_for_DFT_Fx)
        MSE_Fy = MSE(y_true = DFTFy, y_pred = DPMD_for_DFT_Fy)
        MSE_Fz = MSE(y_true = DFTFz, y_pred = DPMD_for_DFT_Fz) 
        MSE_E = MSE(y_true = DFTE, y_pred = DPMD_for_DFT_E) 
        
        file.write(model + " \n")
        file.write("MSE for x force of {} atom_num {} is:".format(config_idx[int(path_idx)],atom_num) +  str(MSE_Fx) + " \n")
        file.write("MSE for y force of {} atom_num {} is:".format(config_idx[int(path_idx)],atom_num) + str(MSE_Fy) + " \n")
        file.write("MSE for z force of {} atom_num {} is:".format(config_idx[int(path_idx)],atom_num) + str(MSE_Fz) + " \n")
        file.write("MSE for energy of {} is:".format(config_idx[int(path_idx)]) +  str(MSE_E) + " \n")
        file.write("\n")
        file.write("\n")
        
        # now plot
        xy_anno1 = (0.25,0.6)
        xy_anno2 = (0.65,0.6)
        xy_anno3 = (0.25,0.15)
        xy_anno4 = (0.6,0.25)

        fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))

        ax1.set_xlabel('MD time step [0.1 fs]')
        ax1.set_ylabel('x Force [eV/A]')
        ax1.annotate("MSE = "+ str(MSE_Fx), xy_anno1, xycoords = "figure fraction")
        ax1.plot(MD_step, DPMDFx, "b", label = "DPMD")
        ax1.plot(np.array(DFT_idx)*100, DPMD_for_DFT_Fx, "go", label = "Target DPMD")
        ax1.plot(np.array(DFT_idx)*100, DFTFx, "ro", label = "DFT")
        
        ax2.set_xlabel('MD time step [0.1 fs]')
        ax2.set_ylabel('y Force [eV/A]')
        ax2.annotate("MSE = "+ str(MSE_Fy), xy_anno2, xycoords = "figure fraction")
        ax2.plot(MD_step, DPMDFy, "b", label = "DPMD")
        ax2.plot(np.array(DFT_idx)*100, DPMD_for_DFT_Fy, "go", label = "Target DPMD")
        ax2.plot(np.array(DFT_idx)*100, DFTFy, "ro", label = "DFT")
        
        ax3.set_xlabel('MD time step [0.1 fs]')
        ax3.set_ylabel('z Force [eV/A]')
        ax3.annotate("MSE = "+ str(MSE_Fz), xy_anno3, xycoords = "figure fraction")
        ax3.plot(MD_step, DPMDFz, "b", label = "DPMD")
        ax3.plot(np.array(DFT_idx)*100, DPMD_for_DFT_Fz, "go", label = "Target DPMD")
        ax3.plot(np.array(DFT_idx)*100, DFTFz, "ro", label = "DFT")
        
        ax4.set_xlabel('MD time step [0.1 fs]')
        ax4.set_ylabel('Energy [eV]')
        ax4.annotate("MSE = "+ str(MSE_E), xy_anno4, xycoords = "figure fraction")
        ax4.plot(MD_step, DPMDE, "b", label = "DPMD")
        ax4.plot(np.array(DFT_idx)*100, DPMD_for_DFT_E, "go", label = "Target DPMD")
        ax4.plot(np.array(DFT_idx)*100, DFTE, "ro", label = "DFT")
        
        plt.title(names[int(path_idx)])
        plt.legend()
        file_name = names[int(path_idx)]+".pdf"
        plt.savefig("DPMDvDFT_graphs/" + file_name)
        plt.close(fig)
        
        
        
def makeMSEPlots(names = None, DFT_file_path = None, \
                 train_index_range = None, valid_index_range = None, test_index_range = None, \
                 bins_train_force = 10, bins_valid_force = 10, bins_test_force = 10,\
                 bins_train_energy = 4, bins_valid_energy = 4, bins_test_energy = 4,\
                 model = None):
    # ambar wants the density of the points, maybe used scatter plots?
    """
    names = names that you want to label the graphs of the parity plots of each system
    DFT_file_path = list containing the file paths to the globbed DFT data
    
    train_index_range = a list of lists containing the [[start_test_idx, end_test_idx],...] of the training set
    valid_index_range = a list of lists containing the [[start_test_idx, end_test_idx],...] of the validation set
    test_index_range = a list of lists containing the [[start_test_idx, end_test_idx],...] of the testing set
    
    model = path to the model file
    """
    working_dir = os.getcwd()

    os.mkdir("./mse_graphs")
    
    keys = []
    for i in range(len(names)):  
        keys.append(i)
        
    config_idx = dict(zip(keys,names))
                          
    
    # write a file that houses all of the MSEs
    file = open("./MSEs.txt","a")
    
    for path_idx in range(len(names)):
        XFx0_train = []
        YFx0_train = []
        XFy0_train = []
        YFy0_train = []
        XFz0_train = []
        YFz0_train = []
        XE_train = []
        YE_train = []
        
        XFx0_valid = []
        YFx0_valid = []
        XFy0_valid = []
        YFy0_valid = []
        XFz0_valid = []
        YFz0_valid = []
        XE_valid = []
        YE_valid = []
        
        XFx0_test = []
        YFx0_test = []
        XFy0_test = []
        YFy0_test = []
        XFz0_test = []
        YFz0_test = []
        XE_test = []
        YE_test = []
        
        
        for i in range(train_index_range[int(path_idx)]): 
            correct = read(DFT_file_path[int(path_idx)] + "/training/dp.traj", index = i)
            XFx0_train.append(correct.get_forces(apply_constraint =False)[:,0])
            XFy0_train.append(correct.get_forces(apply_constraint = False)[:,1])
            XFz0_train.append(correct.get_forces(apply_constraint = False)[:,2])
            XE_train.append(correct.get_potential_energy(apply_constraint = False))

            atoms = read(DFT_file_path[int(path_idx)] + "/training/dp.traj", index = i)
            del atoms.calc
            atoms.calc = DP(model = model) 
            YFx0_train.append(atoms.get_forces()[:,0])
            YFy0_train.append(atoms.get_forces()[:,1])
            YFz0_train.append(atoms.get_forces()[:,2])
            YE_train.append(atoms.get_potential_energy())
            
        for i in range(valid_index_range[int(path_idx)]): 
            correct = read(DFT_file_path[int(path_idx)] + "/validation/dp.traj", index = i)
            XFx0_valid.append(correct.get_forces(apply_constraint =False)[:,0])
            XFy0_valid.append(correct.get_forces(apply_constraint = False)[:,1])
            XFz0_valid.append(correct.get_forces(apply_constraint = False)[:,2])
            XE_valid.append(correct.get_potential_energy(apply_constraint = False))

            atoms = read(DFT_file_path[int(path_idx)] + "/validation/dp.traj", index = i)
            del atoms.calc
            atoms.calc = DP(model = model) 
            YFx0_valid.append(atoms.get_forces()[:,0])
            YFy0_valid.append(atoms.get_forces()[:,1])
            YFz0_valid.append(atoms.get_forces()[:,2])
            YE_valid.append(atoms.get_potential_energy())
            
        for i in range(test_index_range[int(path_idx)]): 
            correct = read(DFT_file_path[int(path_idx)] + "/testing/dp.traj", index = i)
            XFx0_test.append(correct.get_forces(apply_constraint =False)[:,0])
            XFy0_test.append(correct.get_forces(apply_constraint = False)[:,1])
            XFz0_test.append(correct.get_forces(apply_constraint = False)[:,2])
            XE_test.append(correct.get_potential_energy(apply_constraint = False))

            atoms = read(DFT_file_path[int(path_idx)] + "/testing/dp.traj", index = i)
            del atoms.calc
            atoms.calc = DP(model = model) 
            YFx0_test.append(atoms.get_forces()[:,0])
            YFy0_test.append(atoms.get_forces()[:,1])
            YFz0_test.append(atoms.get_forces()[:,2])
            YE_test.append(atoms.get_potential_energy())
            
        # now find the MSE between forces and energies
        # for the final iteration of this code we can make it better by changing it to loop thru all of these
        product_train = (len(XFx0_train))*(len(XFx0_train[0])) # find the len of the reshape matrix
        XFx_train = np.reshape(XFx0_train,[product_train])
        YFx_train = np.reshape(YFx0_train,[product_train])
        XFy_train = np.reshape(XFy0_train,[product_train])
        YFy_train = np.reshape(YFy0_train,[product_train])
        XFz_train = np.reshape(XFz0_train,[product_train])
        YFz_train = np.reshape(YFz0_train,[product_train])
        
        product_valid = (len(XFx0_valid))*(len(XFx0_valid[0])) # find the len of the reshape matrix
        XFx_valid = np.reshape(XFx0_valid,[product_valid])
        YFx_valid = np.reshape(YFx0_valid,[product_valid])
        XFy_valid = np.reshape(XFy0_valid,[product_valid])
        YFy_valid = np.reshape(YFy0_valid,[product_valid])
        XFz_valid = np.reshape(XFz0_valid,[product_valid])
        YFz_valid = np.reshape(YFz0_valid,[product_valid])
        
        product_test = (len(XFx0_test))*(len(XFx0_test[0])) # find the len of the reshape matrix
        XFx_test = np.reshape(XFx0_test,[product_test])
        YFx_test = np.reshape(YFx0_test,[product_test])
        XFy_test = np.reshape(XFy0_test,[product_test])
        YFy_test = np.reshape(YFy0_test,[product_test])
        XFz_test = np.reshape(XFz0_test,[product_test])
        YFz_test = np.reshape(YFz0_test,[product_test])

        MSE_Fx_train = MSE(y_true = XFx_train, y_pred = YFx_train)
        MSE_Fy_train = MSE(y_true = XFy_train, y_pred = YFy_train)
        MSE_Fz_train = MSE(y_true = XFz_train, y_pred = YFz_train) 
        MSE_E_train = MSE(y_true = XE_train, y_pred = YE_train) 
        
        MSE_Fx_valid = MSE(y_true = XFx_valid, y_pred = YFx_valid)
        MSE_Fy_valid = MSE(y_true = XFy_valid, y_pred = YFy_valid)
        MSE_Fz_valid = MSE(y_true = XFz_valid, y_pred = YFz_valid) 
        MSE_E_valid = MSE(y_true = XE_valid, y_pred = YE_valid)
        
        MSE_Fx_test = MSE(y_true = XFx_test, y_pred = YFx_test)
        MSE_Fy_test = MSE(y_true = XFy_test, y_pred = YFy_test)
        MSE_Fz_test = MSE(y_true = XFz_test, y_pred = YFz_test) 
        MSE_E_test = MSE(y_true = XE_test, y_pred = YE_test)
        
        file.write(model + " \n")
        file.write("MSE for x force of {} training is:".format(config_idx[int(path_idx)]) +  str(MSE_Fx_train) + " \n")
        file.write("MSE for y force of {} training is:".format(config_idx[int(path_idx)]) + str(MSE_Fy_train) + " \n")
        file.write("MSE for z force of {} training is:".format(config_idx[int(path_idx)]) + str(MSE_Fz_train) + " \n")
        file.write("MSE for energy of {} training is:".format(config_idx[int(path_idx)]) +  str(MSE_E_train) + " \n")
        file.write("\n")
        
        file.write("MSE for x force of {} validation is:".format(config_idx[int(path_idx)]) +  str(MSE_Fx_valid) + " \n")
        file.write("MSE for y force of {} validation is:".format(config_idx[int(path_idx)]) + str(MSE_Fy_valid) + " \n")
        file.write("MSE for z force of {} validation is:".format(config_idx[int(path_idx)]) + str(MSE_Fz_valid) + " \n")
        file.write("MSE for energy of {} validation is:".format(config_idx[int(path_idx)]) +  str(MSE_E_valid) + " \n")
        file.write("\n")
        
        file.write("MSE for x force of {} testing is:".format(config_idx[int(path_idx)]) +  str(MSE_Fx_test) + " \n")
        file.write("MSE for y force of {} testing is:".format(config_idx[int(path_idx)]) + str(MSE_Fy_test) + " \n")
        file.write("MSE for z force of {} testing is:".format(config_idx[int(path_idx)]) + str(MSE_Fz_test) + " \n")
        file.write("MSE for energy of {} testing is:".format(config_idx[int(path_idx)]) +  str(MSE_E_test) + " \n")
        file.write("\n")
    
        # now plot 
        # now plot 
        xy_anno1 = (0.25,0.6)
        xy_anno2 = (0.65,0.6)
        xy_anno3 = (0.25,0.15)
        xy_anno4 = (0.6,0.25)

        fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))

        ax1.plot(XFx_train,XFx_train, "r--", rasterized=True)
        ax1.set_ylabel('DPD x Force [eV/A]')
        ax1.set_xlabel('DFT x Force [eV/A]')
        ax1.annotate("MSE = "+ str(MSE_Fx_train), xy_anno1, xycoords = "figure fraction")
        hist1,edges1 = np.histogram(YFx_train,bins_train_force)
        c1 = []
        for size1 in hist1:
            for i in range(size1):
                c1.append(size1)        
        zipped1 = zip(XFx_train,YFx_train)
        sorted_zip1 = sorted(zipped1)
        XFx_train_new = []
        YFx_train_new = []
        for t in tuple(sorted_zip1):
            XFx_train_new.append(t[0])
            YFx_train_new.append(t[1])
        sp1 = ax1.scatter(XFx_train_new,YFx_train_new, c=c1, cmap = "jet")
        plt.colorbar(sp1)
        
        ax2.plot(XFy_train,XFy_train, "r--", rasterized=True)
        ax2.set_ylabel('DPD y Force [eV/A]')
        ax2.set_xlabel('DFT y Force [eV/A]')
        ax2.annotate("MSE = "+ str(MSE_Fy_train), xy_anno2, xycoords = "figure fraction")
        hist2,edges2 = np.histogram(YFy_train,bins_train_force)
        c2 = []
        for size2 in hist2:
            for i in range(size2):
                c2.append(size2)
        zipped2 = zip(XFy_train,YFy_train)
        sorted_zip2 = sorted(zipped2)
        XFy_train_new = []
        YFy_train_new = []
        for t in tuple(sorted_zip2):
            XFy_train_new.append(t[0])
            YFy_train_new.append(t[1])
        sp2 = ax2.scatter(XFy_train_new,YFy_train_new, c=c2, cmap = "jet")
        plt.colorbar(sp2)

        ax3.plot(XFz_train,XFz_train, "r--", rasterized=True)
        ax3.set_ylabel('DPD z Force [eV/A]')
        ax3.set_xlabel('DFT z Force [eV/A]')
        ax3.annotate("MSE = "+ str(MSE_Fz_train), xy_anno3, xycoords = "figure fraction")
        hist3,edges3 = np.histogram(YFz_train,bins_train_force)
        c3 = []
        for size3 in hist3:
            for i in range(size3):
                c3.append(size3)    
        zipped3 = zip(XFz_train,YFz_train)
        sorted_zip3 = sorted(zipped3)
        XFz_train_new = []
        YFz_train_new = []
        for t in tuple(sorted_zip3):
            XFz_train_new.append(t[0])
            YFz_train_new.append(t[1])
        sp3 = ax3.scatter(XFz_train_new,YFz_train_new, c=c3, cmap = "jet")
        plt.colorbar(sp3)

        ax4.plot(XE_train,XE_train, "r--", rasterized=True)
        ax4.set_ylabel('DPD Energy [eV]')
        ax4.set_xlabel('DFT Energy [eV]')
        ax4.annotate("MSE = "+ str(MSE_E_train), xy_anno4, xycoords = "figure fraction")
        hist4,edges4 = np.histogram(YE_train,bins_train_energy)
        c4 = []
        for size4 in hist4:
            for i in range(size4):
                c4.append(size4)
        zipped4 = zip(XE_train,YE_train)
        sorted_zip4 = sorted(zipped4)
        XE_train_new = []
        YE_train_new = []
        for t in tuple(sorted_zip4):
            XE_train_new.append(t[0])
            YE_train_new.append(t[1])
        sp4 = ax4.scatter(XE_train_new,YE_train_new, c=c4, cmap = "jet")
        plt.colorbar(sp4)

        plt.title(names[int(path_idx)] + "Training " + model)
        file_name = names[int(path_idx)]+"_training.pdf"
        plt.savefig("mse_graphs/" + file_name)
        plt.close(fig)

        fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))

        ax1.plot(XFx_valid,XFx_valid, "r--", rasterized=True)
        ax1.set_ylabel('DPD x Force [eV/A]')
        ax1.set_xlabel('DFT x Force [eV/A]')
        ax1.annotate("MSE = "+ str(MSE_Fx_valid), xy_anno1, xycoords = "figure fraction")
        hist1,edges1 = np.histogram(YFx_valid,bins_valid_force)
        c1 = []
        for size1 in hist1:
            for i in range(size1):
                c1.append(size1)
        zipped1 = zip(XFx_valid,YFx_valid)
        sorted_zip1 = sorted(zipped1)
        XFx_valid_new = []
        YFx_valid_new = []
        for t in tuple(sorted_zip1):
            XFx_valid_new.append(t[0])
            YFx_valid_new.append(t[1])
        sp1 = ax1.scatter(XFx_valid_new,YFx_valid_new, c=c1, cmap = "jet")
        plt.colorbar(sp1)

        ax2.plot(XFy_valid,XFy_valid, "r--", rasterized=True)
        ax2.set_ylabel('DPD y Force [eV/A]')
        ax2.set_xlabel('DFT y Force [eV/A]')
        ax2.annotate("MSE = "+ str(MSE_Fy_valid), xy_anno2, xycoords = "figure fraction")
        hist2,edges2 = np.histogram(YFy_valid,bins_valid_force)
        c2 = []
        for size2 in hist2:
            for i in range(size2):
                c2.append(size2)
        zipped2 = zip(XFy_valid,YFy_valid)
        sorted_zip2 = sorted(zipped2)
        XFy_valid_new = []
        YFy_valid_new = []
        for t in tuple(sorted_zip2):
            XFy_valid_new.append(t[0])
            YFy_valid_new.append(t[1])
        sp2 = ax2.scatter(XFy_valid_new,YFy_valid_new, c=c2, cmap = "jet")
        plt.colorbar(sp2)

        ax3.plot(XFz_valid,XFz_valid, "r--", rasterized=True)
        ax3.set_ylabel('DPD z Force [eV/A]')
        ax3.set_xlabel('DFT z Force [eV/A]')
        ax3.annotate("MSE = "+ str(MSE_Fz_valid), xy_anno3, xycoords = "figure fraction")
        hist3,edges3 = np.histogram(YFz_valid,bins_valid_force)
        c3 = []
        for size3 in hist3:
            for i in range(size3):
                c3.append(size3)
        zipped3 = zip(XFz_valid,YFz_valid)
        sorted_zip3 = sorted(zipped3)
        XFz_valid_new = []
        YFz_valid_new = []
        for t in tuple(sorted_zip3):
            XFz_valid_new.append(t[0])
            YFz_valid_new.append(t[1])
        sp3 = ax3.scatter(XFz_valid_new,YFz_valid_new, c=c3, cmap = "jet")
        plt.colorbar(sp3)

        ax4.plot(XE_valid,XE_valid, "r--", rasterized=True)
        ax4.set_ylabel('DPD Energy [eV]')
        ax4.set_xlabel('DFT Energy [eV]')
        ax4.annotate("MSE = "+ str(MSE_E_valid), xy_anno4, xycoords = "figure fraction")
        hist4,edges4 = np.histogram(YE_valid,bins_valid_energy)
        c4 = []
        for size4 in hist4:
            for i in range(size4):
                c4.append(size4)  
        zipped4 = zip(XE_valid,YE_valid)
        sorted_zip4 = sorted(zipped4)
        XE_valid_new = []
        YE_valid_new = []
        for t in tuple(sorted_zip4):
            XE_valid_new.append(t[0])
            YE_valid_new.append(t[1])
        sp4 = ax4.scatter(XE_valid_new,YE_valid_new, c=c4, cmap = "jet")
        plt.colorbar(sp4)

        plt.title(names[int(path_idx)] + "Validation " + model)
        file_name = names[int(path_idx)]+"_validation_.pdf"
        plt.savefig("mse_graphs/"+file_name)
        plt.close(fig)
        
        fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))

        ax1.plot(XFx_test,XFx_test, "r--", rasterized=True)
        ax1.set_ylabel('DPD x Force [eV/A]')
        ax1.set_xlabel('DFT x Force [eV/A]')
        ax1.annotate("MSE = "+ str(MSE_Fx_test), xy_anno1, xycoords = "figure fraction")
        hist1,edges1 = np.histogram(YFx_test,bins_test_force)
        c1 = []
        for size1 in hist1:
            for i in range(size1):
                c1.append(size1)
        zipped1 = zip(XFx_test,YFx_test)
        sorted_zip1 = sorted(zipped1)
        XFx_test_new = []
        YFx_test_new = []
        for t in tuple(sorted_zip1):
            XFx_test_new.append(t[0])
            YFx_test_new.append(t[1])
        sp1 = ax1.scatter(XFx_test_new,YFx_test_new, c=c1, cmap = "jet")
        plt.colorbar(sp1)

        ax2.plot(XFy_test,XFy_test, "r--", rasterized=True)
        ax2.set_ylabel('DPD y Force [eV/A]')
        ax2.set_xlabel('DFT y Force [eV/A]')
        ax2.annotate("MSE = "+ str(MSE_Fy_test), xy_anno2, xycoords = "figure fraction")
        hist2,edges2 = np.histogram(YFy_test,bins_test_force)
        c2 = []
        for size2 in hist2:
            for i in range(size2):
                c2.append(size2)
        zipped2 = zip(XFy_test,YFy_test)
        sorted_zip2 = sorted(zipped2)
        XFy_test_new = []
        YFy_test_new = []
        for t in tuple(sorted_zip2):
            XFy_test_new.append(t[0])
            YFy_test_new.append(t[1])
        sp2 = ax2.scatter(XFy_test_new,YFy_test_new, c=c2, cmap = "jet")
        plt.colorbar(sp2)

        ax3.plot(XFz_test,XFz_test, "r--", rasterized=True)
        ax3.set_ylabel('DPD z Force [eV/A]')
        ax3.set_xlabel('DFT z Force [eV/A]')
        ax3.annotate("MSE = "+ str(MSE_Fz_test), xy_anno3, xycoords = "figure fraction")
        hist3,edges3 = np.histogram(YFz_test,bins_test_force)
        c3 = []
        for size3 in hist3:
            for i in range(size3):
                c3.append(size3)
        zipped3 = zip(XFz_test,YFz_test)
        sorted_zip3 = sorted(zipped3)
        XFz_test_new = []
        YFz_test_new = []
        for t in tuple(sorted_zip3):
            XFz_test_new.append(t[0])
            YFz_test_new.append(t[1])
        sp3 = ax3.scatter(XFz_test_new,YFz_test_new, c=c3, cmap = "jet")
        plt.colorbar(sp3)

        ax4.plot(XE_test,XE_test, "r--", rasterized=True)
        ax4.set_ylabel('DPD Energy [eV]')
        ax4.set_xlabel('DFT Energy [eV]')
        ax4.annotate("MSE = "+ str(MSE_E_test), xy_anno4, xycoords = "figure fraction")
        hist4,edges4 = np.histogram(YE_test,bins_test_energy)
        c4 = []
        for size4 in hist4:
            for i in range(size4):
                c4.append(size4)
        zipped4 = zip(XE_test,YE_test)
        sorted_zip4 = sorted(zipped4)
        XE_test_new = []
        YE_test_new = []
        for t in tuple(sorted_zip4):
            XE_test_new.append(t[0])
            YE_test_new.append(t[1])
        sp4 = ax4.scatter(XE_test_new,YE_test_new, c=c4, cmap = "jet")
        plt.colorbar(sp4)

        plt.title(names[int(path_idx)] + "Testing " + model)
        file_name = names[int(path_idx)]+"_testing.pdf"
        plt.savefig("mse_graphs/" + file_name)
        plt.close(fig)

    file.close()
    
    
"""
def makeMSEPlotsTotal(names = None, DFT_file_path = None, \
                 train_index_range = None, valid_index_range = None, test_index_range = None, \
                 bins_train_force = 10, bins_valid_force = 10, bins_test_force = 10,\
                 bins_train_energy = 4, bins_valid_energy = 4, bins_test_energy = 4,\
                 model = None):
    # ambar wants the density of the points, maybe used scatter plots?
    "
    names = names that you want to label the graphs of the parity plots of each system
    DFT_file_path = list containing the file paths to the globbed DFT data
    
    train_index_range = a list of lists containing the [[start_test_idx, end_test_idx],...] of the training set
    valid_index_range = a list of lists containing the [[start_test_idx, end_test_idx],...] of the validation set
    test_index_range = a list of lists containing the [[start_test_idx, end_test_idx],...] of the testing set
    
    model = path to the model file
    "
    working_dir = os.getcwd()

    os.mkdir("./total_mse_graphs")
   
    XF_train_all = []
    YF_train_all = []
    XE_train_all = []
    YE_train_all = []
        
    XF_valid_all = []
    YF_valid_all = []
    XE_valid_all = []
    YE_valid_all = []
        
    XF_test_all = []
    YF_test_all = []
    XE_test_all = []
    YE_test_all = []
    
       
    for path_idx in range(len(DFT_file_path)):
        for i in range(train_index_range[int(path_idx)]): 
                correct = read(DFT_file_path[int(path_idx)] + "/training/dp.traj", index = i)
                XF_train.append(correct.get_forces(apply_constraint =False)[:,0])
                XF_train.append(correct.get_forces(apply_constraint = False)[:,1])
                XF_train.append(correct.get_forces(apply_constraint = False)[:,2])
                XE_train_all.append(correct.get_potential_energy(apply_constraint = False))

                atoms = read(DFT_file_path[int(path_idx)] + "/training/dp.traj", index = i)
                del atoms.calc
                atoms.calc = DP(model = models) 
                YF_train.append(atoms.get_forces()[:,0])
                YF_train.append(atoms.get_forces()[:,1])
                YF_train.append(atoms.get_forces()[:,2])
                YE_train.append(atoms.get_potential_energy())

            for i in range(valid_index_range[int(path_idx)]): 
                correct = read(DFT_file_path[int(path_idx)] + "/validation/dp.traj", index = i)
                XF_valid.append(correct.get_forces(apply_constraint =False)[:,0])
                XF_valid.append(correct.get_forces(apply_constraint = False)[:,1])
                XF_valid.append(correct.get_forces(apply_constraint = False)[:,2])
                XE_valid.append(correct.get_potential_energy(apply_constraint = False))

                atoms = read(DFT_file_path[int(path_idx)] + "/validation/dp.traj", index = i)
                del atoms.calc
                atoms.calc = DP(model = models)
                YF_valid.append(atoms.get_forces()[:,0])
                YF_valid.append(atoms.get_forces()[:,1])
                YF_valid.append(atoms.get_forces()[:,2])
                YE_valid.append(atoms.get_potential_energy())

            for i in range(test_index_range[int(path_idx)]): 
                correct = read(DFT_file_path[int(path_idx)] + "/testing/dp.traj", index = i)
                XF_test.append(correct.get_forces(apply_constraint =False)[:,0])
                XF_test.append(correct.get_forces(apply_constraint = False)[:,1])
                XF_test.append(correct.get_forces(apply_constraint = False)[:,2])
                XE_test.append(correct.get_potential_energy(apply_constraint = False))

                atoms = read(DFT_file_path[int(path_idx)] + "/testing/dp.traj", index = i)
                del atoms.calc
                atoms.calc = DP(model = models) 
                YF_test.append(atoms.get_forces()[:,0])
                YF_test.append(atoms.get_forces()[:,1])
                YF_test.append(atoms.get_forces()[:,2])
                YE_test.append(atoms.get_potential_energy())
        
      
        MSE_F_train = MSE(y_true = XF_train, y_pred = YF_train) 
        MSE_E_train = MSE(y_true = XE_train, y_pred = YE_train) 

        MSE_F_valid = MSE(y_true = XF_valid, y_pred = YF_valid)
        MSE_E_valid = MSE(y_true = XE_valid, y_pred = YE_valid)

        MSE_F_test = MSE(y_true = XF_test, y_pred = YF_test)
        MSE_E_test = MSE(y_true = XE_test, y_pred = YE_test)
        
        # now plot 
        xy_anno1 = (0.7,0.3)
        
        # training
        plt.plot(XF_train,XF_train, "r--", rasterized=True)
        plt.set_ylabel('DPD Force [eV/A]')
        plt.set_xlabel('DFT Force [eV/A]')
        plt.annotate("MSE = "+ str(MSE_F_train), xy_anno1, xycoords = "figure fraction")
        hist1,edges1 = np.histogram(YF_train,bins_train_force)
        c1 = []
        for size1 in hist1:
            for i in range(size1):
                c1.append(size1)        
        zipped1 = zip(XF_train,YF_train)
        sorted_zip1 = sorted(zipped1)
        XF_train_new = []
        YF_train_new = []
        for t in tuple(sorted_zip1):
            XF_train_new.append(t[0])
            YF_train_new.append(t[1])
        sp1 = plt.scatter(XF_train_new,YF_train_new, c=c1, cmap = "jet")
        plt.colorbar(sp1)
        file_name = "training_force.pdf"
        plt.savefig("total_mse_graphs/" + file_name)
        plt.close(fig)
        
        # validation
        plt.plot(XF_valid,XF_valid, "r--", rasterized=True)
        plt.set_ylabel('DPD Force [eV/A]')
        plt.set_xlabel('DFT Force [eV/A]')
        plt.annotate("MSE = "+ str(MSE_F_valid), xy_anno1, xycoords = "figure fraction")
        hist1,edges1 = np.histogram(YF_valid,bins_valid_force)
        c1 = []
        for size1 in hist1:
            for i in range(size1):
                c1.append(size1)
        zipped1 = zip(XF_valid,YF_valid)
        sorted_zip1 = sorted(zipped1)
        XF_valid_new = []
        YF_valid_new = []
        for t in tuple(sorted_zip1):
            XF_valid_new.append(t[0])
            YF_valid_new.append(t[1])
        sp1 = plt.scatter(XF_valid_new,YF_valid_new, c=c1, cmap = "jet")
        plt.colorbar(sp1)
        file_name = "validation_force.pdf"
        plt.savefig("total_mse_graphs/"+file_name)
        plt.close(fig)
        
        # testing
        plt.plot(XF_test,XF_test, "r--", rasterized=True)
        plt.set_ylabel('DPD Force [eV/A]')
        plt.set_xlabel('DFT Force [eV/A]')
        plt.annotate("MSE = "+ str(MSE_F_test), xy_anno1, xycoords = "figure fraction")
        hist1,edges1 = np.histogram(YF_test,bins_test_force)
        c1 = []
        for size1 in hist1:
            for i in range(size1):
                c1.append(size1)
        zipped1 = zip(XF_test,YF_test)
        sorted_zip1 = sorted(zipped1)
        XF_test_new = []
        YF_test_new = []
        for t in tuple(sorted_zip1):
            XF_test_new.append(t[0])
            YF_test_new.append(t[1])
        sp1 = plt.scatter(XF_test_new,YF_test_new, c=c1, cmap = "jet")
        plt.colorbar(sp1)
        file_name = "testing_force.pdf"
        plt.savefig("total_mse_graphs/" + file_name)
        plt.close(fig)
        

        # training

        plt.plot(XE_train,XE_train, "r--", rasterized=True)
        plt.set_ylabel('DPD Energy [eV]')
        plt.set_xlabel('DFT Energy [eV]')
        plt.annotate("MSE = "+ str(MSE_E_train), xy_anno1, xycoords = "figure fraction")
        hist1,edges1 = np.histogram(YE_train,bins_train_energy)
        c1 = []
        for size1 in hist1:
            for i in range(size1):
                c1.append(size1)        
        zipped1 = zip(XE_train,YE_train)
        sorted_zip1 = sorted(zipped1)
        XE_train_new = []
        YE_train_new = []
        for t in tuple(sorted_zip1):
            XE_train_new.append(t[0])
            YE_train_new.append(t[1])
        sp1 = plt.scatter(XE_train_new,YE_train_new, c=c1, cmap = "jet")
        plt.colorbar(sp1)
        file_name = "training_energy.pdf"
        plt.savefig("total_mse_graphs/" + file_name)
        plt.close(fig)

        # validation
        plt.plot(XE_valid,XE_valid, "r--", rasterized=True)
        plt.set_ylabel('DPD Energy [eV]')
        plt.set_xlabel('DFT Energy [eV]')
        plt.annotate("MSE = "+ str(MSE_E_valid), xy_anno1, xycoords = "figure fraction")
        hist1,edges1 = np.histogram(YE_valid,bins_valid_energy)
        c1 = []
        for size1 in hist1:
            for i in range(size1):
                c1.append(size1)
        zipped1 = zip(XE_valid,YE_valid)
        sorted_zip1 = sorted(zipped1)
        XE_valid_new = []
        YE_valid_new = []
        for t in tuple(sorted_zip1):
            XE_valid_new.append(t[0])
            YE_valid_new.append(t[1])
        sp1 = plt.scatter(XE_valid_new,YE_valid_new, c=c1, cmap = "jet")
        plt.colorbar(sp1)
        file_name = "validation_energy.pdf"
        plt.savefig("total_mse_graphs/"+file_name)
        plt.close(fig)
        
        # testing
        plt.plot(XE_test,XE_test, "r--", rasterized=True)
        plt.set_ylabel('DPD Energy [eV/A]')
        plt.set_xlabel('DFT Energy [eV/A]')
        plt.annotate("MSE = "+ str(MSE_E_test), xy_anno1, xycoords = "figure fraction")
        hist1,edges1 = np.histogram(YE_test,bins_test_energy)
        c1 = []
        for size1 in hist1:
            for i in range(size1):
                c1.append(size1)
        zipped1 = zip(XE_test,YE_test)
        sorted_zip1 = sorted(zipped1)
        XE_test_new = []
        YE_test_new = []
        for t in tuple(sorted_zip1):
            XE_test_new.append(t[0])
            YE_test_new.append(t[1])
        sp1 = plt.scatter(XE_test_new,YE_test_new, c=c1, cmap = "jet")
        plt.colorbar(sp1)
        file_name = "testing_energy.pdf"
        plt.savefig("total_mse_graphs/" + file_name)
        plt.close(fig)
"""


def makeMSEBarChart_old(names = None, DFT_file_path = None, \
                 train_index_range = None, valid_index_range = None, test_index_range = None, \
                 model = None):

    """
    names = names of each config
    DFT_file_path = list containing the file paths to the globbed DFT data

    train_index_range = a list of lists containing the [[start_test_idx, end_test_idx],...] of the training set
    valid_index_range = a list of lists containing the [[start_test_idx, end_test_idx],...] of the validation set
    test_index_range = a list of lists containing the [[start_test_idx, end_test_idx],...] of the testing set

    model = path to the model file
    """
    os.mkdir("bar_charts")

    working_dir = os.getcwd()
    pdfs = []

    # write a file that houses all of the MSEs
    """
    MSE_Fx_train = []
    MSE_Fy_train = []
    MSE_Fz_train = []
    MSE_E_train = []

    MSE_Fx_valid = []
    MSE_Fy_valid = []
    MSE_Fz_valid = []
    MSE_E_valid = []
    """
    
    MSE_Fx_test = []
    MSE_Fy_test = []
    MSE_Fz_test = []
    MSE_E_test = []


    for path_idx in range(len(names)):
        """
        XFx0_train = []
        YFx0_train = []
        XFy0_train = []
        YFy0_train = []
        XFz0_train = []
        YFz0_train = []
        XE_train = []
        YE_train = []

        XFx0_valid = []
        YFx0_valid = []
        XFy0_valid = []
        YFy0_valid = []
        XFz0_valid = []
        YFz0_valid = []
        XE_valid = []
        YE_valid = []
        """
        
        XFx0_test = []
        YFx0_test = []
        XFy0_test = []
        YFy0_test = []
        XFz0_test = []
        YFz0_test = []
        XE_test = []
        YE_test = []

        """
        for i in range(train_index_range[int(path_idx)]):
            correct = read(DFT_file_path[int(path_idx)] + "/training/dp.traj", index = i)
            XFx0_train.append(correct.get_forces(apply_constraint =False)[:,0])
            XFy0_train.append(correct.get_forces(apply_constraint = False)[:,1])
            XFz0_train.append(correct.get_forces(apply_constraint = False)[:,2])
            XE_train.append(correct.get_potential_energy(apply_constraint = False))

            atoms = read(DFT_file_path[int(path_idx)] + "/training/dp.traj", index = i)
            del atoms.calc
            atoms.calc = DP(model = model)
            YFx0_train.append(atoms.get_forces()[:,0])
            YFy0_train.append(atoms.get_forces()[:,1])
            YFz0_train.append(atoms.get_forces()[:,2])
            YE_train.append(atoms.get_potential_energy())

        for i in range(valid_index_range[int(path_idx)]):
            correct = read(DFT_file_path[int(path_idx)] + "/validation/dp.traj", index = i)
            XFx0_valid.append(correct.get_forces(apply_constraint =False)[:,0])
            XFy0_valid.append(correct.get_forces(apply_constraint = False)[:,1])
            XFz0_valid.append(correct.get_forces(apply_constraint = False)[:,2])
            XE_valid.append(correct.get_potential_energy(apply_constraint = False))

            atoms = read(DFT_file_path[int(path_idx)] + "/validation/dp.traj", index = i)
            del atoms.calc
            atoms.calc = DP(model = model)
            YFx0_valid.append(atoms.get_forces()[:,0])
            YFy0_valid.append(atoms.get_forces()[:,1])
            YFz0_valid.append(atoms.get_forces()[:,2])
            YE_valid.append(atoms.get_potential_energy())
            """

        for i in range(test_index_range[int(path_idx)]):
            correct = read(DFT_file_path[int(path_idx)] + "/testing/dp.traj", index = i)
            XFx0_test.append(correct.get_forces(apply_constraint =False)[:,0])
            XFy0_test.append(correct.get_forces(apply_constraint = False)[:,1])
            XFz0_test.append(correct.get_forces(apply_constraint = False)[:,2])
            XE_test.append(correct.get_potential_energy(apply_constraint = False))

            atoms = read(DFT_file_path[int(path_idx)] + "/testing/dp.traj", index = i)
            del atoms.calc
            atoms.calc = DP(model = model)
            YFx0_test.append(atoms.get_forces()[:,0])
            YFy0_test.append(atoms.get_forces()[:,1])
            YFz0_test.append(atoms.get_forces()[:,2])
            YE_test.append(atoms.get_potential_energy())
        """
        # now find the MSE between forces and energies
        # for the final iteration of this code we can make it better by changing it to loop thru all of these
        product_train = (len(XFx0_train))*(len(XFx0_train[0])) # find the len of the reshape matrix
        XFx_train = np.reshape(XFx0_train,[product_train])
        YFx_train = np.reshape(YFx0_train,[product_train])
        XFy_train = np.reshape(XFy0_train,[product_train])
        YFy_train = np.reshape(YFy0_train,[product_train])
        XFz_train = np.reshape(XFz0_train,[product_train])
        YFz_train = np.reshape(YFz0_train,[product_train])

        product_valid = (len(XFx0_valid))*(len(XFx0_valid[0])) # find the len of the reshape matrix
        XFx_valid = np.reshape(XFx0_valid,[product_valid])
        YFx_valid = np.reshape(YFx0_valid,[product_valid])
        XFy_valid = np.reshape(XFy0_valid,[product_valid])
        YFy_valid = np.reshape(YFy0_valid,[product_valid])
        XFz_valid = np.reshape(XFz0_valid,[product_valid])
        YFz_valid = np.reshape(YFz0_valid,[product_valid])
        """

        product_test = (len(XFx0_test))*(len(XFx0_test[0])) # find the len of the reshape matrix
        XFx_test = np.reshape(XFx0_test,[product_test])
        YFx_test = np.reshape(YFx0_test,[product_test])
        XFy_test = np.reshape(XFy0_test,[product_test])
        YFy_test = np.reshape(YFy0_test,[product_test])
        XFz_test = np.reshape(XFz0_test,[product_test])
        YFz_test = np.reshape(YFz0_test,[product_test])
        
        """
        MSE_Fx_train.append(MSE(y_true = XFx_train, y_pred = YFx_train))
        MSE_Fy_train.append(MSE(y_true = XFy_train, y_pred = YFy_train))
        MSE_Fz_train.append(MSE(y_true = XFz_train, y_pred = YFz_train))
        MSE_E_train.append(MSE(y_true = XE_train, y_pred = YE_train))

        MSE_Fx_valid.append(MSE(y_true = XFx_valid, y_pred = YFx_valid))
        MSE_Fy_valid.append(MSE(y_true = XFy_valid, y_pred = YFy_valid))
        MSE_Fz_valid.append(MSE(y_true = XFz_valid, y_pred = YFz_valid))
        MSE_E_valid.append(MSE(y_true = XE_valid, y_pred = YE_valid))
        """
        
        MSE_Fx_test.append(MSE(y_true = XFx_test, y_pred = YFx_test))
        MSE_Fy_test.append(MSE(y_true = XFy_test, y_pred = YFy_test))
        MSE_Fz_test.append(MSE(y_true = XFz_test, y_pred = YFz_test))
        MSE_E_test.append(MSE(y_true = XE_test, y_pred = YE_test))
    """
    dict_Fx_train = {names:MSE_Fx_train for names,MSE_Fx_train in zip(names,MSE_Fx_train)}
    dict_Fy_train = {names:MSE_Fy_train for names,MSE_Fy_train in zip(names,MSE_Fy_train)}
    dict_Fz_train = {names:MSE_Fz_train for names,MSE_Fz_train in zip(names,MSE_Fz_train)}
    dict_E_train = {names:MSE_E_train for names,MSE_E_train in zip(names,MSE_E_train)}

    dict_Fx_valid = {names:MSE_Fx_valid for names,MSE_Fx_valid in zip(names,MSE_Fx_valid)}
    dict_Fy_valid = {names:MSE_Fy_valid for names,MSE_Fy_valid in zip(names,MSE_Fy_valid)}
    dict_Fz_valid = {names:MSE_Fz_valid for names,MSE_Fz_valid in zip(names,MSE_Fz_valid)}
    dict_E_valid = {names:MSE_E_valid for names,MSE_E_valid in zip(names,MSE_E_valid)}
    """
    dict_Fx_test = {names:MSE_Fx_test for names,MSE_Fx_test in zip(names,MSE_Fx_test)}
    dict_Fy_test = {names:MSE_Fy_test for names,MSE_Fy_test in zip(names,MSE_Fy_test)}
    dict_Fz_test = {names:MSE_Fz_test for names,MSE_Fz_test in zip(names,MSE_Fz_test)}
    dict_E_test = {names:MSE_E_test for names,MSE_E_test in zip(names,MSE_E_test)}
    """
    keys_Fx_train = list(dict_Fx_train.keys())
    values_Fx_train = list(dict_Fx_train.values())
    keys_Fy_train = list(dict_Fy_train.keys())
    values_Fy_train = list(dict_Fy_train.values())
    keys_Fz_train = list(dict_Fz_train.keys())
    values_Fz_train = list(dict_Fz_train.values())
    keys_E_train = list(dict_E_train.keys())
    values_E_train = list(dict_E_train.values())
    
    #fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))

    ax1.bar(range(len(dict_Fx_train)), values_Fx_train, rotation = "vertical")
    ax1.set_xticklabels(keys_Fx_train)
    ax.set_ylabel("x Force MSE")
    plt.
    
    ax2.bar(range(len(dict_Fy_train)), values_Fy_train, rotation = "veritcal")
    ax2.set_xticklabels(keys_Fy_train)
    ax2.set_ylabel("y Force MSE")
    
    ax3.bar(range(len(dict_Fz_train)), values_Fz_train, rotation = "vertical")
    ax3.set_xticklabels(keys_Fz_train)
    ax3.set_ylabel("z Force MSE")
    
    ax4.bar(range(len(dict_E_train)), values_E_train, rotation = "vertical")
    ax4.set_xticklabels(keys_E_train)
    ax4.set_ylabel("Energy MSE")
    
    plt.title("Training")
    file_name = "./bar_charts/training.pdf"
    plt.savefig(file_name)
    plt.close(fig)

    pdf_name = file_name 

    pdfs.append(pdf_name)
        
    os.chdir(working_dir)

    keys_Fx_valid = list(dict_Fx_valid.keys())
    values_Fx_valid = list(dict_Fx_valid.values())
    keys_Fy_valid =  list(dict_Fy_valid.keys())
    values_Fy_valid = list(dict_Fy_valid.values())
    keys_Fz_valid = list(dict_Fz_valid.keys())
    values_Fz_valid = list(dict_Fz_valid.values())
    keys_E_valid = list(dict_E_valid.keys())
    values_E_valid = list(dict_E_valid.values())

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))

    ax1.bar(range(len(dict_Fx_valid)), values_Fx_valid, rotation = "vertical")
    ax1.set_xticklabels(keys_Fx_valid)
    ax1.set_ylabel("x Force MSE")
    
    ax2.bar(range(len(dict_Fy_valid)), values_Fy_valid, rotation = "vertical")
    ax2.set_xticklabels(keys_Fy_valid)
    ax2.set_ylabel("y Force MSE")
    
    ax3.bar(range(len(dict_Fz_valid)), values_Fz_valid, rotation = "vertical")
    ax3.set_xticklabels(keys_Fz_valid)
    ax3.set_ylabel("z Force MSE")
    
    ax4.bar(range(len(dict_E_valid)), values_E_valid, rotation = "vertical")
    ax4.set_xticklabels(keys_E_valid)
    ax4.set_ylabel("Energy MSE")
    
    plt.title("Validation")
    file_name = "./bar_charts/validation.pdf"
    plt.savefig(file_name)
    plt.close(fig)

    pdf_name = file_name

    pdfs.append(pdf_name)

    os.chdir(working_dir)
    """
    keys_Fx_test = list(dict_Fx_test.keys())
    values_Fx_test = list(dict_Fx_test.values())
    keys_Fy_test = list(dict_Fy_test.keys())
    values_Fy_test = list(dict_Fy_test.values())
    keys_Fz_test = list(dict_Fz_test.keys())
    values_Fz_test = list(dict_Fz_test.values())
    keys_E_test = list(dict_E_test.keys())
    values_E_test = list(dict_E_test.values())

    #fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))
    # need to fix this bc the plots plotted on top of each other
    # change it so that the MSE for the forces are averaged
    # change the label as well
    fig = plt.Figure(figsize = (20,10))
    plt.bar(range(len(dict_Fx_test)), values_Fx_test)
    plt.xticks(range(len(dict_Fx_test)),keys_Fx_test,rotation = "vertical")
    plt.ylabel("x Force MSE")
    file_name = "./bar_charts/testing_Fx.pdf"
    plt.savefig(file_name)
    plt.close(fig)
    
    fig = plt.Figure(figsize = (10,5))
    plt.bar(range(len(dict_Fy_test)), values_Fy_test)
    plt.xticks(range(len(dict_Fy_test)),keys_Fy_test,rotation = "vertical")
    plt.ylabel("y Force MSE")
    file_name = "./bar_charts/testing_Fy.pdf"
    plt.savefig(file_name)
    plt.close(fig)
    
    fig = plt.Figure(figsize = (10,5))
    plt.bar(range(len(dict_Fz_test)), values_Fz_test)
    plt.xticks(range(len(dict_Fz_test)),keys_Fz_test,rotation = "vertical")
    plt.ylabel("z Force MSE")
    file_name = "./bar_charts/testing_Fz.pdf"
    plt.savefig(file_name)
    plt.close(fig)
    
    fig = plt.Figure(figsize = (10,5))
    plt.bar(range(len(dict_E_test)), values_E_test)
    plt.xticks(range(len(dict_E_test)),keys_E_test,rotation = "vertical")
    plt.ylabel("Energy MSE")
    file_name = "./bar_charts/testing_E.pdf"
    plt.savefig(file_name)
    plt.close(fig)
    
    """
    pdf_name = file_name

    pdfs.append(pdf_name)

    os.chdir(working_dir)

    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(pdf)

    merger.write("bar_charts/bar_chart.pdf")
    merger.close()
    """
    
def validity_Au_distances(paths = None, n_Au = None):
    # run this three times, one for each number of Aus
    # looked at some number from: https://www.researchgate.net/figure/Equilibrium-geometries-at-T-0-K-for-Au-2-Au-3-and-Au-4-and-their-complexes-with-one_fig2_256971302

    # looking at the larger data, the numbers Au-Au distance depends on geometry
    # however on average 2.65A looks close
    print("For n_Au = " + str(n_Au))

    eq_length = 2.65*np.ones((n_Au,n_Au)) #[A]
    # this should be a square matrix of 2.65 with lengths n_Au

    DPMD_lengths = []
    # try to thread this
    for path in paths: # chooses the zeolite
        for j in range(90,100): # chooses the specific image
            # extract the Au
            atoms = read(path+"/output.traj",index = j)[-n_Au:]
            DPMD_lengths.append(atoms.get_all_distances())

    # now check the MSE between the eq_length and the difference DPMD_lengths
    # THIS DOES NOT QUITE DO WAHT WE WANT IT PRINTS FOR EACH STEP OF THE WAY
    MSE_length = []
    for i in range(len(DPMD_lengths)):
        MSE_length.append(MSE(y_pred = DPMD_lengths[i], y_true = eq_length))
        print(MSE_length)
        print()
        
def make_std_hist_old(names = None, paths = None, graphs = None, hist_folder = None, final_pdf = None):
    os.mkdir(hist_folder)

    working_dir = os.getcwd()
    pdfs = []
    colors = sns.color_palette('deep')

    type_map = {'O': 0, 'Si': 1, 'Au': 2} # CHECK THIS

    for i,path in enumerate(paths):
        allatoms = read(path, index=':')

        pos = np.array([atoms.get_positions().flatten() for atoms in allatoms])
        cell = np.array([atoms.cell.array.flatten() for atoms in allatoms])
        types = [type_map[a.symbol] for a in allatoms[0]]

        graphs = graphs

        models = [DeepPot(g) for g in graphs]

        #dev = calc_model_devi(pos, cell, types, models, nopbc=False)

        dev = calc_model_devi(pos, cell, types, models, nopbc=False)[:, 4]

        #plt.hist(dev, 20)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.kdeplot(dev, color=colors[0], fill=True)

        # "ghost" lines for legend
        ax.plot([], [], lw=2.0, color=colors[0])
        plt.title(path)
        ax.set_xlabel('$\epsilon$$_t$', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.set_xlim([0.00, 0.40])
        plt.tick_params(labelsize=12)
        plt.savefig(working_dir + "/hist" + names[i] + ".pdf")
        plt.show()
        plt.close()

        pdf_name = working_dir +"_"+ names[i] + ".pdf"

        pdfs.append(pdf_name)

        os.chdir(working_dir)
    """
    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(pdf)

    merger.write(final_pdf)
    merger.close()
    
    os.system("mv *pdf " + hist_folder)
    """
    
def std_vs_largest_cluster_distance(paths = None, graphs = None, hist_folder = None, final_pdf = None):
    os.mkdir(hist_folder)

    working_dir = os.getcwd()
    pdfs = []

    type_map = {'O': 0, 'Si': 1, 'Au': 2} # CHECK THIS

    
    for i,path in enumerate(paths):
        max_distance = []
        allatoms = read(path, index=':')

        pos = np.array([atoms.get_positions().flatten() for atoms in allatoms])
        cell = np.array([atoms.cell.array.flatten() for atoms in allatoms])
        types = [type_map[a.symbol] for a in allatoms[0]]

        graphs = graphs

        models = [DeepPot(g) for g in graphs]

        #dev = calc_model_devi(pos, cell, types, models, nopbc=False)

        dev = calc_model_devi(pos, cell, types, models, nopbc=False)[:, 4]
        
        # do something with get_all_distances to get the max distance

        #plt.plot()
        plt.title(path)
        plt.savefig(working_dir + "/hist" + names[i] + ".pdf")
        plt.show()
        plt.close()

        pdf_name = working_dir +"_"+ names[i] + ".pdf"

        pdfs.append(pdf_name)

        os.chdir(working_dir)

    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(pdf)

    merger.write(final_pdf)
    merger.close()
    
    
def add_water(x_bounds = None, y_bounds = None, z_bounds = None, traj_to_append = None, water_to_add = None):
    # initialize the diffs so that the while loop runs
    water_added = 0
    print(water_added)
    while water_added != water_to_add:
        x_pos = random.uniform(x_bounds[0],x_bounds[1])
        y_pos = random.uniform(y_bounds[0],y_bounds[1])
        z_pos = random.uniform(z_bounds[0],z_bounds[1])

        h2o = molecule("H2O")
        h2o[0].position = [x_pos,y_pos,z_pos]

        traj_to_append = traj_to_append + h2o[0]

        indexs = []
        for i in range(len(traj_to_append) - 1): # avoids the new O
            indexs.append(i)

        distances = traj_to_append.get_distances(a = indexs, indices = -1,mic = True)

        if distances.flatten().min()<1:
            traj_to_append.pop(-1)

        else:
            diff_01 = [0,0.763,-0.596] # this is the top H (#1)
            diff_02 = [0,-0.763,-0.596]

            h2o[1].position = [x_pos + diff_01[0], y_pos + diff_01[1], z_pos + diff_01[2]]
            h2o[2].position = [x_pos + diff_02[0], y_pos + diff_02[1], z_pos + diff_02[2]]

            traj_to_append = traj_to_append + h2o[1]
            traj_to_append = traj_to_append + h2o[2]
            print(water_added)
            water_added +=1


    return traj_to_append    

def add_OH_Cs(x_bounds = None, y_bounds = None, z_bounds = None, traj_to_append = None):
    oh_added = 0
    oh_to_add = 1
    print(oh_added)
    while oh_added != oh_to_add:
        x_pos = random.uniform(x_bounds[0],x_bounds[1])
        y_pos = random.uniform(y_bounds[0],y_bounds[1])
        z_pos = random.uniform(z_bounds[0],z_bounds[1])

        oh = molecule("OH")
        oh[0].position = [x_pos,y_pos,z_pos]

        traj_to_append = traj_to_append + oh[0]

        indexs = []
        for i in range(len(traj_to_append) - 1): # avoids the new O
            indexs.append(i)

        distances = traj_to_append.get_distances(a = indexs, indices = -1,mic = True)

        if distances.flatten().min()<1:
            traj_to_append.pop(-1)

        else:
            diff_01 = [0,0,-0.979] # this is the top H (#1)

            oh[1].position = [x_pos + diff_01[0], y_pos + diff_01[1], z_pos + diff_01[2]]

            traj_to_append = traj_to_append + oh[1]
            print(oh_added)
            oh_added +=1
            
    cs_added = 0
    cs_to_add = 1
    print(cs_added)
    while cs_added != cs_to_add:
        x_pos = random.uniform(x_bounds[0],x_bounds[1])
        y_pos = random.uniform(y_bounds[0],y_bounds[1])
        z_pos = random.uniform(z_bounds[0],z_bounds[1])

        cs = Atom("Cs", position = [x_pos,y_pos,z_pos])

        traj_to_append = traj_to_append + cs
        indexs = []
        for i in range(len(traj_to_append) - 1): # avoids the new O
            indexs.append(i)

        distances = traj_to_append.get_distances(a = indexs, indices = -1,mic = True)

        if distances.flatten().min()<3:
            traj_to_append.pop(-1)

        else:
            print(cs_added)
            cs_added +=1

    return traj_to_append

def random_seed():
    if len(sys.argv) > 1:
        dp_files = sys.argv[1:] 
    else:
        dp_files = ['in.json']
        
    for f in dp_files:
        with open(f) as file:
            dp_in = json.loads(file.read())
        seed1 = np.random.randint(99999)
        seed2 = np.random.randint(99999)
        seed3 = np.random.randint(99999)

        dp_in['model']['descriptor']['seed'] = seed1
        dp_in['model']['fitting_net']['seed'] = seed2
        dp_in['training']['seed'] = seed3

        with open(f, 'w') as file:
            file.write(json.dumps(dp_in, indent=2))
            
def set_up_training(path = None, num_config_train = None, num_config_valid = None, apply_constraint = False):
    
    #TODO: figure out how to run the raw_to_set.sh in a different directory, and figure out why to code after the raw_to_set runs before it
    
    """
    path = the path into each folder housing an out put from the DFT {}_{}/{}_{}/file_name
    num_config = the number of configurations total that are being trained
    save_path = the location that the box.raw, etc are to be saved
    apply_constraint = whether to apply the constraint on the DFT forces and energies
    
    size_of_set = the number of trajs per set
    set_to_move = the number of the set to be used for testing
    
    """    
    working_directory = os.getcwd() # when in submission loop, the cwd is the folders that you are submitted i
    
    MakeBoxRaw(path = path, num_config_train = num_config_train, num_config_valid = num_config_valid)
    MakeCoordRaw(path = path, num_config_train = num_config_train, num_config_valid = num_config_valid)
    MakeEnergyRaw(path = path, num_config_train = num_config_train, num_config_valid = num_config_valid, apply_constraint = False)
    MakeForceRaw(path = path, num_config_train = num_config_train, num_config_valid = num_config_valid, apply_constraint = False)
    MakeVirialRaw(path = path, num_config_train = num_config_train, num_config_valid = num_config_valid)
    MakeTypeRaw(path = path)
    
    # training
    os.system("mv " + path + "/training/box.raw " + working_directory)
    os.system("mv " + path + "/coord.raw " + working_directory)
    os.system("mv " + path + "/energy.raw " + working_directory)
    os.system("mv " + path + "/force.raw " + working_directory)
    os.system("mv " + path + "/type.raw " + working_directory)
    os.system("mv " + path + "/virial.raw " + working_directory)

    # run raw_to_set.sh
    os.system("source ./raw_to_set.sh 10")
    
    os.system("mv ./box.raw " + path)
    os.system("mv ./coord.raw " + path)
    os.system("mv ./energy.raw " + path)
    os.system("mv ./force.raw " + path)
    os.system("mv ./type.raw " + path)
    os.system("mv ./virial.raw " + path)
    
    os.system("mv ./set.0* " + path + "/training")
    
    # validation
    os.system("mv " + path + "/validation/box.raw " + working_directory)
    os.system("mv " + path + "/validation/coord.raw " + working_directory)
    os.system("mv " + path + "/validation/energy.raw " + working_directory)
    os.system("mv " + path + "/validation/force.raw " + working_directory)
    os.system("mv " + path + "/validation/type.raw " + working_directory)
    os.system("mv " + path + "/validation/virial.raw " + working_directory)

    # run raw_to_set.sh
    os.system("source ./raw_to_set.sh 10")
    
    os.system("mv ./box.raw " + path + "/validation")
    os.system("mv ./coord.raw " + path + "/validation")
    os.system("mv ./energy.raw " + path + "/validation")
    os.system("mv ./force.raw " + path +"/validation")
    os.system("mv ./type.raw " + path + "/validation")
    os.system("mv ./virial.raw " + path + "/validation")
    
    os.system("mv ./set.0* " + path + "/validation")





def transfer_DFT_for_DPMD(iters = ["01", "02", "03", "04"], num1 = None, zeo = None, num2 = None, metal = None):
    #THIS ASSUME THAT THE COMBINED DFT FILES ARE CALLED DFT.TRAJ
    # WE ALSO NEED TO MAKE ALL OF THE DATA INSIDE TRAINING, VALIDATION, AND TESTING TO BE CALLED DP.TRAJ
    for i in iters:
        for n1,z in zip(num1,zeo):
            for n2,m in zip(num2,metal):
                os.system("cp ./data/{}_{}/{}_{}/DFT.traj ./{}/{}_{}/{}_{}".format(n1,z,n2,m,i,n1,z,n2,m))
                
def make_train_valid_test_split(num_train = None, num_valid = None):
    all_atoms = read("DFT.traj", index = ":")
    train = all_atoms[num_train:]
    valid = all_atoms[num_train:(num_train+num_valid)]
    test = all_atoms[(num_train+num_valid):len(all_atoms)]
    
    write("./training/dp.traj",train)
    write("./validation/dp.traj",valid)
    write("./testing/dp.traj",test)
    
def find_best_model(DFT_file_path = None, index_range = None, models = None, models_name = None):
    """
    DFT_file_path = list containing the file paths to the globbed DFT data
    index_range = a list of lists containing the [[start_test_idx, end_test_idx],...]
    model = list to all of the models to check (do not include the directory it is houed in (this is specified in models_name))
    models_name = list of the folders where the models are housed
    """

    file = open("./total_MSEs.txt", "a")

    for model, model_name in zip(models, models_name):
        total_MSE = 0.

        os.chdir("./{}".format(model_name))

        for path_idx in range(len(DFT_file_path)):
            XFx0 = []
            YFx0 = []
            XFy0 = []
            YFy0 = []
            XFz0 = []
            YFz0 = []
            XE = []
            YE = []

            #make

            # now find the MSE between forces and energies
            # for the final iteration of this code we can make it better by changing it to loop thru all of these
            product = (len(XFx0))*(len(XFx0[0])) # find the len of the reshape matrix
            XFx = np.reshape(XFx0,[product])
            YFx = np.reshape(YFx0,[product])
            XFy = np.reshape(XFy0,[product])
            YFy = np.reshape(YFy0,[product])
            XFz = np.reshape(XFz0,[product])
            YFz = np.reshape(YFz0,[product])

            MSE_Fx = MSE(y_true = XFx, y_pred = YFx)
            MSE_Fy = MSE(y_true = XFy, y_pred = YFy)
            MSE_Fz = MSE(y_true = XFz, y_pred = YFz)
            MSE_E = MSE(y_true = XE, y_pred = YE)

            total_MSE += MSE_Fx
            total_MSE += MSE_Fy
            total_MSE += MSE_Fz
            total_MSE += MSE_E

        file.write("MSE total for {} is:".format(model_name) +  str(total_MSE) + " \n")
        file.write("\n")

        os.chdir("../")

    file.close()
    
def make_validationDPMD_folders(iters = ["01", "02", "03", "04"], num1 = None, zeo = None, num2 = None, metal = None):
    for i in iters:
        for n1,z in zip(num1,zeo):
            os.mkdir(path = "./{}/{}_{}/".format(i,n1,z))
            for n2,m in zip(num2,metal):
                os.mkdir(path = "./{}/{}_{}/{}_{}/".format(i,n1,z,n2,m))
                
                
def run_SinglePointEnergy(path =  None,vasp_parameters = {'gga':'RP',"encut":400, "nsw":0},atom_type = None,
                          use_db = False, database = None, path_to_DPMD = None,
                          zeo = None, method = None, iteration = None, n_metal = None, metal = None, idx = None):
    # The reason this code is failing is because when we run this it is in the specific job folder, which means a specific index
    # howver the current updateDFT takes all of the values, so it fails
    # one way to get the index is to use the folder's last number and using this sraw the number from the DFT_write
    
    """
    (all paths are with respect to the job folder that the code will be run in)
    path = the path to the atoms to have DFT run on it 
    vasp_parameters = the vasp parameters to be used in the calculation
    atom_type = the type of atoms used in the calcualtion for kT to add other defaults into vasp parameters (for kT)
    
    database = the database to update
    use_db = tells if the db should be used for not
    path_to_DPMD = the specific {}_{}/{}_{}/{whatever the file was named} that we will take the DP MD from
    currently_in_DFT = wheter or not the Atoms object in the db is in the process of having DFT run on it
    path = the specific {}_{}/{}_{}/{whatever the file was named} that we will take the DP MD from 
    zeo = the spcific zeolite we are considering
    method = the method that these configurations were saved
    iteration = the iteraation number that this file was generated in
    n_metal = the number of metal atoms inside the trajectory
    metal = the chemical symbol of the metal
    (always throw some away from the start of the MD to avoid the part going from 0K to temeprature K)
    idx = the index of the trajectory to be updated
    """
    # run the code to set the database value as running
    if use_db == True:
        updateDFTStatus(database = database, currently_in_DFT = True, path = path_to_DPMD, zeo = zeo, method = method,\
                        iteration = iteration, n_metal = n_metal, metal = metal, idx = idx)
    
    kt = KT(gamma_only=False,structure_type=str(atom_type))
    kt.set_calculation_type('opt')
    atoms = read(path)
    
    atoms.pbc=True
    kt.set_structure(atoms)
    kt.set_overall_vasp_params(vasp_parameters)
    kt.run()
     # run the code to set the database code as complete
    if use_db == True:
        updateDFTStatus(database = database, currently_in_DFT = False, path = path_to_DPMD, zeo = zeo, method = method,\
                    iteration = iteration, n_metal = n_metal, metal = metal, idx = idx)