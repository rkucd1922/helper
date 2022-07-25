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

def AddDPMDTODatabase(database = None, path_to_data = None, zeo = None, method = "DPMD",\
                     iteration = None, n_metal = None, metal = None, num_to_save = None, num_images_to_toss = None,
                     path_to_save_txt = None): 
    
    """
    database = the database to save to
    path_to_data = the path to the DPMD data we will be saving
    zeo = the spcific zeolite we are considering
    method = the method that these configurations were saved
    iteration = the iteraation number that this file was generated in
    n_metal = the number of metal atoms inside the trajectory
    metal = the chemical symbol of the metal
    num_to_save = the number of images from the DPMD.traj to save in the database
    num_images_to_toss = the number of images to toss 
    (always throw some away from the start of the MD to avoid the part going from 0K to temeprature K)
    path_to_save_txt = the path to save the txt file that will contain the information about the DPMD trajs added to the database
    """
    
    
    db = connect(database)
    
    file = open(path_to_save_txt ,"a")
    file.write(str(metal) + " " + str(n_metal) + " " + str(zeo) + " " + method + " " + str(iteration))
    file.write(" \n ")
    
    all_atoms_total = read(path_to_data, index = str(num_images_to_toss) + ":")
    # this accounts for the images we will toss (allows for interval of 1)
    
    if num_to_save > len(all_atoms_total):
        raise ValueError("Can not have the number of images to save larger than the total atoms put into the system")
    
    images = []
    interval = int(np.floor(len(all_atoms_total)/num_to_save))
    for i in range(num_to_save):
        idx = i*interval + num_images_to_toss
        images.append(idx)
            
    all_atoms = []
    for i in range(len(images)):
        all_atoms.append(read(path_to_data, index = images[i]))
        
    for i, atoms in enumerate(all_atoms):
        db.write(atoms, metal = metal, idx = images[i], n_metal = n_metal, zeo = zeo, method = method, iteration = iteration, \
                 DFT_status = "Not Run")
        file.write(str(images[i]))
        file.write(", ")
    
    file.write(" \n ")
    file.write(" \n ")
    file.close()

def AddDFTToDatabase(database = None, zeo = None, method = "DFT_SPE",\
                     iteration = None, n_metal = None, metal = None):
    # change this to be run in the 00{} folder
    """
    database = the database to save to
    path = the path to the 00{} folders that house all of the DFT SPE vasprun.xml data we will be saving
    zeo = the spcific zeolite we are considering
    method = the method that these configurations were saved
    iteration = the iteraation number that this file was generated in
    n_metal = the number of metal atoms inside the trajectory
    metal = the chemical symbol of the metal
    """
    
    db = connect(database)
    
    atoms = read("./opt_PBE_400_111/vasprun.xml", index = -1) # this is to get the vasprun.xml
    current_directory = os.getcwd()[-3:] 
    
    i = int(current_directory[2:])
    db.write(atoms, metal = metal, idx = i, n_metal = n_metal, zeo = zeo, method = method, iteration = iteration, \
                     DFT_status = "Complete")
        
        
def AddVASPMDToDatabase(database = None, path = None, zeo = None, method = "DFT_MD",\
                     iteration = None, n_metal = None, metal = None):
    """
    database = the database to save to
    path = the specific path to the VASP_MD run
    zeo = the spcific zeolite we are considering
    method = the method that these configurations were saved
    iteration = the iteraation number that this file was generated in
    n_metal = the number of metal atoms inside the trajectory
    metal = the chemical symbol of the metal
    """
    
    db = connect(database)
    
    
    all_atoms = read(path, index = ":")
    for i, atoms in enumerate(all_atoms):
        db.write(atoms, metal = metal, idx = i, n_metal = n_metal, zeo = zeo, method = method, \
                     iteration = iteration, DFT_status = "Completed")


def updateDFTStatus(database = None, currently_in_DFT = False, path = None, zeo = None, method = "DPMD",\
                     iteration = None, n_metal = None, metal = None, idx = None):
    """
    database = the database to update
    currently_in_DFT = wheter or not the Atoms object in the db is in the process of having DFT run on it
    path = the specific {}_{}/{}_{}/{whatever the file was named} that we will take the DP MD from
    zeo = the spcific zeolite we are considering
    method = the method that these configurations were saved
    iteration = the iteraation number that this file was generated in
    n_metal = the number of metal atoms inside the trajectory
    metal = the chemical symbol of the metal
    idx = the index of the trajectory to be updated
    """
    # mod this to take a specific index
    
    db = connect(database)
    
    if currently_in_DFT == True:
        id = db.get(DFT_status = "Not Run", metal = metal, n_metal = n_metal, idx = idx, \
                    zeo = zeo, method = method, iteration = iteration).id 
        db.update(id, DFT_status = "Running") 
        
    
    if currently_in_DFT == False: 
        id = db.get(DFT_status = "Running", metal = metal, n_metal = n_metal, idx = idx,\
                    zeo = zeo, method = method, iteration = iteration).id 
        db.update(id,DFT_status = "Complete")