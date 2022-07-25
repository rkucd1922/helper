import os, sys 
sys.path.append("./../")
from helper.io import combinePDFs
from ase.io import read, write
import numpy as np
import glob
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileMerger


def check_conv(folder):
    # this still need to pick up on if the DFT crashed for another reason
    """
    folder = the DFT folder that contains the OSZICAR to be checked
    """
    
    with open(folder + "/OSZICAR") as file:
        line = [l for l in file if 'RMM' in l][-1]
    n = int(line.split()[1])
    if n == 60:
        print(folder, 'unconverged!!')
        return False
        
    file = open(folder + "/OSZICAR")
    last_line = file.readlines()[-1]
    if str(last_line[:2]) == "RMM":  # this doesnt work
        print(folder, 'unconverged!!')
        return False
    
    else:
        return True
    
def combineVASPruns(DFT_folder_name = None):
    """
    DFT_folder_name = the folder that contains the DFT run
    """
    
    all_atoms = []
    all_folders = glob.glob("./0*/" + DFT_folder_name)
    
    for folder in all_folders:
        if check_conv(folder) == True:
            all_atoms.append(read(folder + "/vasprun.xml"))
    
    write("./DFT.traj", all_atoms)
    
def check_kpts_lattice_constant(path = None, kpts = None, true_volume = None, true_cell = None):
    # rn this only works if the lattice constant is of the form [[x,0,0],[0,x,0],[0,0,x]] # must be fixed
    """
    path = path to one of the DFT runs so that the number of atoms can be found
    kpts = the list of tested k points
    true_volume = the volume to be used as the standard
    true_cell = the unit cell to be used as the standard
    """
    all_f = glob.glob("0*/opt*")
    all_f = sorted(all_f)

    kpts = kpts

    true_volume_array = np.zeros(len(kpts))
    true_cell_array = np.zeros(len(kpts))
    for i in range(len(true_volume)):
        true_volume_array[i] = true_volume
        true_cell_array[i] = true_cell

    atoms = read(path)
    len_atoms = len(atoms)

    volume = []
    cell_0 = [] # since the unit cell experimentally is of the form [[x,0,0],[0,x,0],[0,0,x]] i will only consider the x values
    cell_1 = []
    cell_2 = []
    for f in all_f:
        atoms = read(f + "/vasprun.xml", index = -1)
        volume.append(atoms.get_volume()/len_atoms)
        cell_0.append(atoms.get_cell()[0][0])
        cell_1.append(atoms.get_cell()[1][1])
        cell_2.append(atoms.get_cell()[2][2])

    plt.plot(kpts,volume,"bo")
    plt.plot(kpts,true_volume_array,"r--")
    plt.ylabel("Cell Volume [A^3]")
    plt.xlabel("kpts")
    plt.savefig("volume.pdf")
    plt.close()

    plt.plot(kpts,cell_0,"bo")
    plt.plot(kpts,true_cell_array,"r--")
    plt.ylabel("Cell Lenght (Axis 0) [A]")
    plt.xlabel("kpts")
    plt.savefig("cell_0.pdf")
    plt.close()

    plt.plot(kpts,cell_1,"bo")
    plt.plot(kpts,true_cell_array,"r--")
    plt.ylabel("Cell Lenght (Axis 1) [A]")
    plt.xlabel("kpts")
    plt.savefig("cell_1.pdf")
    plt.close()

    plt.plot(kpts,cell_2,"bo")
    plt.plot(kpts,true_cell_array,"r--")
    plt.ylabel("Cell Lenght (Axis 2) [A]")
    plt.xlabel("kpts")
    plt.savefig("cell_2.pdf")
    plt.close()
    
    pdfs = ["volume.pdf", "cell_0.pdf", "cell_1.pdf", "cell_2.pdf"]
    combinePDFs(pdfs = pdfs, pdf_name = "kpts.pdf")
    
def check_encut_lattice_constant(path = None, encuts = None, true_volume = None, true_cell = None):
    # rn this only works if the lattice constant is of the form [[x,0,0],[0,x,0],[0,0,x]] # must be fixed
    """
    path = path to one of the DFT runs so that the number of atoms can be found
    enucts = the list of tested k points
    true_volume = the volume to be used as the standard
    true_cell = the unit cell to be used as the standard
    """
    all_f = glob.glob("0*/opt*")
    all_f = sorted(all_f)

    encut = encut
    true_volume = np.zeros(len(encuts))
    true_cell = np.zeros(len(encuts))
    for i in range(len(true_volume)):
        true_volume[i] = 11.82
        true_cell[i] = 3.615

    atoms = read(path, index = -1)
    len_atoms = len(atoms)
    
    volume = []
    cell_0 = []
    cell_1 = []
    cell_2 = []
    for f in all_f:
        atoms = read(f + "/vasprun.xml", index = -1)
        volume.append(atoms.get_volume()/len_atoms)
        cell_0.append(atoms.get_cell()[0][0])
        cell_1.append(atoms.get_cell()[1][1])
        cell_2.append(atoms.get_cell()[2][2])

    plt.plot(encut,volume,"bo")
    plt.plot(encut,true_volume_array,"r--")
    plt.ylabel("Cell Volume [A^3]")
    plt.xlabel("Encut [eV]")
    plt.savefig("volume.pdf")
    plt.close()

    plt.plot(encut,cell_0,"bo")
    plt.plot(encut,true_cell_array,"r--")
    plt.ylabel("Cell Lenght (Axis 0) [A]")
    plt.xlabel("Encut [eV]")
    plt.savefig("cell_0.pdf")
    plt.close()

    plt.plot(encut,cell_1,"bo")
    plt.plot(encut,true_cell_array,"r--")
    plt.ylabel("Cell Lenght (Axis 1) [A]")
    plt.xlabel("Encut [eV]")
    plt.savefig("cell_1.pdf")
    plt.close()

    plt.plot(encut,cell_2,"bo")
    plt.plot(encut,true_cell_array,"r--")
    plt.ylabel("Cell Lenght (Axis 2) [A]")
    plt.xlabel("Encut [eV]")
    plt.savefig("cell_2.pdf")
    plt.close()
    
    pdfs = ["volume.pdf", "cell_0.pdf", "cell_1.pdf", "cell_2.pdf"]
    combinePDFs(pdfs = pdfs, pdf_name = "encuts.pdf")