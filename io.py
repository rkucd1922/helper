import os, sys 
from ase.io import read, write
import glob
from PyPDF2 import PdfFileMerger


def extract(path = None, num_to_save = None, num_images_to_toss = None):
    """
    path = the path to the atoms
    num_to_save = the number of images from the DPMD.traj to save in the database
    num_images_to_toss = the number of images of the DPMD to toss (from database) 
    (always throw some away from the start of the MD to avoid the part going from 0K to temeprature K)
    """
    
    all_atoms_total = read(path, index = str(num_images_to_toss) + ":") 
    # this accounts for the images we will toss (allows for interval of 1)
    
    if num_to_save > len(all_atoms_total):
        raise ValueError("Can not have the number of images to save larger than the total atoms put into the system")
    
    interval = int(np.floor(len(all_atoms_total)/num_to_save))
    
    atoms_list = []
    for i in range(num_to_save):
        idx = (i*interval + num_images_to_toss)
        atoms_list.append(read(path, index = idx))
        
    return atoms_list

def combinePDFs(pdfs = None, pdf_name = None):
    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(pdf)

    merger.write(pdf_name)
    merger.close()

def combine_traj(paths_combined = None, save_path_combined = None):
    """
    paths_combined = the list of paths to the trajs to be combined
    save_path_combined = path to save the combined trajectory (must include the name of the new trajectory)
    """
    trajs = []
    len_combined_trajs = []
    for path in paths_combined:
        atoms = read(path, index = ":")
        trajs.append(atoms)
        len_combined_trajs.append(len(atoms))
    
    total_len = sum(len_combined_trajs)
        
    traj = trajs[0]
    for t in range(1,len(trajs)):
        traj += trajs[t]

    write(save_path_combined, traj)
    return total_len

def find_lowest_energy(path = None):
    allatoms = read(path,index = ":")
    energy = []
    for atoms in allatoms:
        energy.append(atoms.get_potential_energy())
    lowest_energy = min(energy)  
    lowest_energy_index = energy.index(lowest_energy)
    return lowest_energy_index


def make_folders_DPMD(iteration = None, num1 = None, zeo = None, num2 = None, metal_list = None):
    """
    paths = the paths of the metal/zeolite combinations that this iteration will train on (form of {}_{}/{}_{})
    iteration = iteration number (also name of folder that contains the {}_{}/{}_{})
    additional_path = path to place between the current folder and the {}_{}/{}_{} folders (start with /<path>)"""
    os.mkdir("iteration_{}".format(iteration))
    os.mkdir("iteration_{}/data".format(iteration))
    for n1,z in zip(num1,zeo):
        os.mkdir("./iteration_{}/{}_{}".format(iteration,n1,z))
        for n2,m in zip(num2,metal_list):
            os.mkdir("./iteration_{}/data/{}_{}/{}_{}".format(iteration,n1,z,n2,m))
            os.mkdir("." + additional_path + "/iteration_{}/data/{}_{}/{}_{}/training".format(iteration,n1,z,n2,m))
            os.mkdir("." + additional_path + "/iteration_{}/data/{}_{}/{}_{}/validation".format(iteration,n1,z,n2,m))

def make_DFT_folders(path = None, num_jobs = None):
    """the path to a folder that will need job folders added in it
    num_jobs = the number of jobs to be run"""
    
    str_num_jobs = str(num_jobs)
    characters = len(num_jobs)
    
    for i in range(num_jobs):
        os.mkdir(path + "/" + str(i).zfill(int(characters+1)))