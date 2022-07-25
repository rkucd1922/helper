import os, sys 
sys.path.append("./../")
from helper.io import combinePDFs
from ase.io import read, write
import numpy as np
from deepmd.calculator import DP
from deepmd.infer import calc_model_devi
from deepmd.infer import DeepPot
from deepmd.infer.model_devi import calc_model_devi_v, calc_model_devi_f
import glob
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileMerger
import seaborn as sns

def get_dev(atoms = None, type_map = None, graphs = None):
    """
    atoms = the atoms that are going to be checked for model deviations
    type_map = chemical symbol type dictionary (ex:{'O': 0, 'Si': 1, 'Au': 2})
    graphs = list containing the models that we will be checking the deviations between
    """
    
    pos = np.array([a.get_positions().flatten() for a in atoms])
    cell = np.array([a.cell.array.flatten() for a in atoms])
    types = [type_map[a.symbol] for a in atoms[0]]

    models = [DeepPot(g) for g in graphs]

    dev = calc_model_devi(pos, cell, types, models, nopbc=False)[:, 4]
    return dev 
    
def makeDPMDvDFTPlots(names = None, DPMD_data = None, DFT_file_path = None, DFT_idx = None, model = None, atom_num = None):
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
         
        DFTFx_all = []
        DFTFy_all = []
        DFTFz_all = []
        DPMDFx_all = []
        DPMDFy_all = []
        DPMDFz_all = []
        
        
        # append the DPMD data
        DPMD = read(DPMD_data[int(path_idx)], index = ":")
        for i,atoms in enumerate(DPMD):
            DPMDFx.append(atoms.get_forces()[:,0][atom_num])
            DPMDFy.append(atoms.get_forces()[:,1][atom_num])
            DPMDFz.append(atoms.get_forces()[:,2][atom_num])
            DPMDE.append(atoms.get_potential_energy())
            MD_step.append(i*100)
            
        DPMD_using_DFT_Fx = []
        DPMD_using_DFT_Fy = []
        DPMD_using_DFT_Fz = []
        DPMD_using_DFT_E = []
            
        DFT = read(DFT_file_path[int(path_idx)], index = ":")
        for atoms in DFT:
            DFTFx.append(atoms.get_forces()[:,0][atom_num])
            DFTFy.append(atoms.get_forces()[:,1][atom_num])
            DFTFz.append(atoms.get_forces()[:,2][atom_num])
            DFTE.append(atoms.get_potential_energy())
            
            DFTFx_all.append(atoms.get_forces()[:,0])
            DFTFy_all.append(atoms.get_forces()[:,1])
            DFTFz_all.append(atoms.get_forces()[:,2])
        
            
            del atoms.calc
            atoms.calc = DP(model = model)
            DPMD_using_DFT_Fx.append(atoms.get_forces()[:,0][atom_num])
            DPMD_using_DFT_Fy.append(atoms.get_forces()[:,1][atom_num])
            DPMD_using_DFT_Fz.append(atoms.get_forces()[:,2][atom_num])
            DPMD_using_DFT_E.append(atoms.get_potential_energy())
            
            DPMDFx_all.append(atoms.get_forces()[:,0])
            DPMDFy_all.append(atoms.get_forces()[:,1])
            DPMDFz_all.append(atoms.get_forces()[:,2])
            
        np.array(DFTFx_all).flatten()
        np.array(DFTFy_all).flatten()
        np.array(DFTFz_all).flatten()
        np.array(DPMDFx_all).flatten()
        np.array(DPMDFy_all).flatten()
        np.array(DPMDFz_all).flatten()
            
        
        MSE_Fx = MSE(y_true = DFTFx_all, y_pred = DPMDFx_all)
        MSE_Fy = MSE(y_true = DFTFy_all, y_pred = DPMDFy_all)
        MSE_Fz = MSE(y_true = DFTFz_all, y_pred = DPMDFz_all) 
        MSE_E = MSE(y_true = DFTE, y_pred = DPMD_using_DFT_E) 
        
        file.write(model + " \n")
        file.write("MSE for x force of {} is:".format(config_idx[int(path_idx)]) +  str(MSE_Fx) + " \n")
        file.write("MSE for y force of {} is:".format(config_idx[int(path_idx)]) + str(MSE_Fy) + " \n")
        file.write("MSE for z force of {} is:".format(config_idx[int(path_idx)]) + str(MSE_Fz) + " \n")
        file.write("MSE for energy of {} is:".format(config_idx[int(path_idx)]) +  str(MSE_E) + " \n")
        file.write("\n")
        file.write("\n")
        
        # now plot
        #xy_anno1 = (0.25,0.6)
        #xy_anno2 = (0.65,0.6)
        #xy_anno3 = (0.25,0.15)
        #xy_anno4 = (0.6,0.25)

        fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))

        ax1.set_xlabel('MD time step [0.1 fs]')
        ax1.set_ylabel('x Force [eV/A]')
        #ax1.annotate("MSE = "+ str(MSE_Fx), xy_anno1, xycoords = "figure fraction")
        ax1.plot(MD_step, DPMDFx, "b", label = "DPMD")
        ax1.plot(np.array(DFT_idx)*100, DPMD_using_DFT_Fx, "go", label = "DPD calc")
        ax1.plot(np.array(DFT_idx)*100, DFTFx, "ro", label = "DFT")
        
        ax2.set_xlabel('MD time step [0.1 fs]')
        ax2.set_ylabel('y Force [eV/A]')
        #ax2.annotate("MSE = "+ str(MSE_Fy), xy_anno2, xycoords = "figure fraction")
        ax2.plot(MD_step, DPMDFy, "b", label = "DPMD")
        ax2.plot(np.array(DFT_idx)*100, DPMD_using_DFT_Fy, "go", label = "DPD calc")
        ax2.plot(np.array(DFT_idx)*100, DFTFy, "ro", label = "DFT")
        
        ax3.set_xlabel('MD time step [0.1 fs]')
        ax3.set_ylabel('z Force [eV/A]')
        #ax3.annotate("MSE = "+ str(MSE_Fz), xy_anno3, xycoords = "figure fraction")
        ax3.plot(MD_step, DPMDFz, "b", label = "DPMD")
        ax3.plot(np.array(DFT_idx)*100, DPMD_using_DFT_Fz, "go", label = "DPD calc")
        ax3.plot(np.array(DFT_idx)*100, DFTFz, "ro", label = "DFT")
        
        ax4.set_xlabel('MD time step [0.1 fs]')
        ax4.set_ylabel('Energy [eV]')
        #ax4.annotate("MSE = "+ str(MSE_E), xy_anno4, xycoords = "figure fraction")
        ax4.plot(MD_step, DPMDE, "b", label = "DPMD")
        ax4.plot(np.array(DFT_idx)*100, DPMD_using_DFT_E, "go", label = "DPD calc")
        ax4.plot(np.array(DFT_idx)*100, DFTE, "ro", label = "DFT")
        
        plt.title(names[int(path_idx)])
        plt.legend()
        file_name = names[int(path_idx)]+".pdf"
        plt.savefig("DPMDvDFT_graphs/" + file_name)
        plt.close(fig)
        
def makeMSEBarChart(names = None, DFT_file_path = None, \
                 train_index_range = None, valid_index_range = None, test_index_range = None, \
                 model = None):
    # i want this to run for all iterations with the iterations plotted on top of each other
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
    
    MSE_F_test = []
    MSE_E_test = []
    
    iterations = range(len(model))


    for path_idx in range(len(names)):     
        for iteration in range(len(model)):
            
            XF0_test = []
            YF0_test = []
            YFz0_test = []
            XE_test = []
            YE_test = []
            
            empty_list = []
            MSE_F_test.append(empty_list)
            
            empty_list = []
            MSE_E_test.append(empty_list)

            for i in range(test_index_range[iteration][int(path_idx)]):
                correct = read(DFT_file_path[iteration][int(path_idx)] + "/testing/dp.traj", index = i)
                XF0_test.append(correct.get_forces(apply_constraint =False))
                XE_test.append(correct.get_potential_energy(apply_constraint = False))

                atoms = read(DFT_file_path[iteration][int(path_idx)] + "/testing/dp.traj", index = i)
                del atoms.calc
                atoms.calc = DP(model = model[iteration])
                YF0_test.append(atoms.get_forces())
                YE_test.append(atoms.get_potential_energy())

            XF_test = np.array(XF0_test).flatten()
            YF_test = np.array(YF0_test).flatten()

            MSE_F_test.append(MSE(y_true = XF_test, y_pred = YF_test))
            MSE_E_test.append(MSE(y_true = XE_test, y_pred = YE_test))
            
    
    i1_F = {names:MSE_F_test for names[0],MSE_F_test[0] in zip(names[0],MSE_F_test[0])}
    i2_F = {names:MSE_F_test for names[1],MSE_F_test[1] in zip(names[1],MSE_F_test[1])}
    i3_F = {names:MSE_F_test for names[2],MSE_F_test[2] in zip(names[2],MSE_F_test[2])}
    i4_F = {names:MSE_F_test for names[3],MSE_F_test[3] in zip(names[3],MSE_F_test[3])}
    i5_F = {names:MSE_F_test for names[4],MSE_F_test[4] in zip(names[4],MSE_F_test[4])}
    
    i1_E = {names:MSE_E_test for names[0],MSE_E_test[0] in zip(names[0],MSE_E_test[0])}
    i2_E = {names:MSE_E_test for names[0],MSE_E_test[0] in zip(names[0],MSE_E_test[0])}
    i3_E = {names:MSE_E_test for names[0],MSE_E_test[0] in zip(names[0],MSE_E_test[0])}
    i4_E = {names:MSE_E_test for names[0],MSE_E_test[0] in zip(names[0],MSE_E_test[0])}
    i5_E = {names:MSE_E_test for names[0],MSE_E_test[0] in zip(names[0],MSE_E_test[0])}
    
    iF = [i1_F,i2_F,i3_F,i4_F,i5_F]
    iE = [i1_E,i2_E,i3_E,i4_E,i5_E]
    
    dict_Fx_test = {iterations[i]:iF[i] for i in iterations}
    dict_E_test = {iterations[i]:iE[i] for i in iterations}

    keys_F_test = []
    values_F_test = [] 
    keys_E_test = []
    values_E_test = []
    
    for iteration in range(len(model)):
        keys_F_test.append(list(dict_F_test[iteration].keys()))
        values_F_test.append(list(dict_F_test[iteration].values()))
        keys_E_test.append(list(dict_E_test[iteration].keys()))
        values_E_test.append(list(dict_E_test[iteration].values()))

    #fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))
    # need to fix this bc the plots plotted on top of each other
    # change it so that the MSE for the forces are averaged
    # change the label as well
    fig = plt.Figure(figsize = (20,10))
    for iteration in range(len(model)):
        plt.bar(range(len(dict_F_test[iteration])), values_F_test[iteration])
        plt.xticks(range(len(dict_F_test[iteration])),keys_F_test[iteration],rotation = "vertical")
    plt.ylabel("x Force MSE")
    file_name = "./bar_charts/testing_F.pdf"
    plt.savefig(file_name)
    plt.close(fig)
    plt.show()
    
    fig = plt.Figure(figsize = (10,5))
    for iteration in range(len(model)):
        plt.bar(range(len(dict_E_test[iteration])), values_E_test[iteration])
        plt.xticks(range(len(dict_E_test[iteration])),keys_E_test[iteration],rotation = "vertical")
    plt.ylabel("Energy MSE")
    file_name = "./bar_charts/testing_E.pdf"
    plt.savefig(file_name)
    plt.close(fig)
    plt.show()
    
def justTestMSEs(names = None, DFT_file_path = None, index_range = None, model = None):
    """
    names = names that you want to label the graphs of the parity plots of each system
    DFT_file_path = list containing the file paths to the globbed DFT data
    index_range = a list of lists containing the [[start_test_idx, end_test_idx],...]
    model = path to the model file
    """
    working_dir = os.getcwd()
    pdfs = []
    
    os.mkdir("./mse_graphs")

    keys = []
    for i in range(len(names)):
        keys.append(i)

    config_idx = dict(zip(keys,names))


    # write a file that houses all of the MSEs
    file = open("./MSEs.txt","a")

    for path_idx in range(len(names)):
        XFx0 = []
        YFx0 = []
        XFy0 = []
        YFy0 = []
        XFz0 = []
        YFz0 = []
        XE = []
        YE = []
        
        for i in range(index_range[path_idx],index_range[path_idx]): # takes the final 10
        # the model will test the data from set.009 (DFT data) then del the calc and use the iteration's calc to check the E and F
        #GET THIS TO FIT THE MAIN.DB MODEL
            correct = read(DFT_file_path[path_idx], index = i)
            XFx0.append(correct.get_forces(apply_constraint =False)[:,0])
            XFy0.append(correct.get_forces(apply_constraint = False)[:,1])
            XFz0.append(correct.get_forces(apply_constraint = False)[:,2])
            XE.append(correct.get_potential_energy(apply_constraint = False))

            atoms = read(DFT_file_path[path_idx],index = i)
            del atoms.calc
            atoms.calc = DP(model = model)
            YFx0.append(atoms.get_forces()[:,0])
            YFy0.append(atoms.get_forces()[:,1])
            YFz0.append(atoms.get_forces()[:,2])
            YE.append(atoms.get_potential_energy())

        # now find the MSE between forces and energies
        # for the final iteration of this code we can make it better by changing it to loop thru all of these
        #product = (len(XFx0))*(len(XFx0[0])) # find the len of the reshape matrix
        XFx = np.array(XFx0).flatten()
        YFx = np.array(YFx0).flatten()
        XFy = np.array(XFy0).flatten()
        YFy = np.array(YFy0).flatten()
        XFz = np.array(XFz0).flatten()
        YFz = np.array(YFz0).flatten()


        MSE_Fx = MSE(y_true = XFx, y_pred = YFx)
        MSE_Fy = MSE(y_true = XFy, y_pred = YFy)
        MSE_Fz = MSE(y_true = XFz, y_pred = YFz)
        MSE_E = MSE(y_true = XE, y_pred = YE)
        
        file.write("MSE for x force of {} is:".format(config_idx[path_idx]) +  str(MSE_Fx) + " \n")
        file.write("MSE for y force of {} is:".format(config_idx[path_idx]) + str(MSE_Fy) + " \n")
        file.write("MSE for z force of {} is:".format(config_idx[path_idx]) + str(MSE_Fz) + " \n")
        file.write("MSE for energy of {} is:".format(config_idx[path_idx]) +  str(MSE_E) + " \n")
        file.write("\n")
        
        # now plot 
        xy_anno1 = (0.25,0.6)
        xy_anno2 = (0.65,0.6)
        xy_anno3 = (0.25,0.15)
        xy_anno4 = (0.6,0.25)

        fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))

        ax1.plot(XFx,YFx, "bo")
        ax1.plot(XFx,XFx, "r--")
        ax1.set_ylabel('DPD x Force [eV/A]')
        ax1.set_xlabel('DFT x Force [eV/A]')
        ax1.annotate("MSE = "+ str(MSE_Fx), xy_anno1, xycoords = "figure fraction")

        ax2.plot(XFy,YFy, "bo")
        ax2.plot(XFy,XFy, "r--")
        ax2.set_ylabel('DPD y Force [eV/A]')
        ax2.set_xlabel('DFT y Force [eV/A]')
        ax2.annotate("MSE = "+ str(MSE_Fy), xy_anno2, xycoords = "figure fraction")

        ax3.plot(XFz,YFz, "bo")
        ax3.plot(XFz,XFz, "r--")
        ax3.set_ylabel('DPD z Force [eV/A]')
        ax3.set_xlabel('DFT z Force [eV/A]')
        ax3.annotate("MSE = "+ str(MSE_Fz), xy_anno3, xycoords = "figure fraction")

        ax4.plot(XE[-1],YE[-1], "bo")
        ax4.plot(XE[-1],XE[-1], "r--")
        ax4.set_ylabel('DPD Energy [eV]')
        ax4.set_xlabel('DFT Energy [eV]')
        ax4.annotate("MSE = "+ str(MSE_E), xy_anno4, xycoords = "figure fraction")

        plt.title(names[path_idx])
        file_name = names[path_idx]+".pdf"
        plt.savefig(file_name)
        plt.close(fig)

        pdf_name = working_dir + "/" + file_name

        pdfs.append(pdf_name)
        
        os.chdir(working_dir)

    file.close()

    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(pdf)

    merger.write("final.pdf")
    merger.close()
    
def make_std_hist_all(name = None, paths = None, graphs = None):
    # i think what I'm going to do is run this in its own folder in the parent folder
    file = open("num_good_total.txt","w")
    type_map = {'O': 0, 'Si': 1, 'Au': 2} # CHECK THIS

    devs = []
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    colors = sns.color_palette('deep')
    for i in range(len(graphs)):
        good = []
        path = paths[i]
        allatoms = read(path, index=':')

        pos = np.array([atoms.get_positions().flatten() for atoms in allatoms])
        cell = np.array([atoms.cell.array.flatten() for atoms in allatoms])
        types = [type_map[a.symbol] for a in allatoms[0]]

        gs = graphs[i]

        models = [DeepPot(g) for g in gs]

        devs.append(calc_model_devi(pos, cell, types, models, nopbc=False)[:, 4]) # appends for each iterations
        
        for j,val in enumerate(devs[-1]):
            if val <= 0.1:
                good.append(j)

        file.write(path + " " +  str(len(good)))
        file.write("\n")

        sns.kdeplot(devs[i], color=colors[i], fill=True)

        # "ghost" lines for legend
        label = "Iteration_" + str(i+1)
        ax.plot([], [], lw=2.0, color=colors[i], label=label)

    ax.set_xlabel('$\epsilon$$_t$', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_xlim([0.00, 0.40])
    plt.tick_params(labelsize=12)
    plt.legend()
    plt.title(name)
    plt.savefig(name + ".pdf")
    file.close()
    
def make_std_hist(name = None, path = None, graphs = None):
    # i think what I'm going to do is run this in its own folder in the parent folder
    file = open("num_good_total.txt","w")
    type_map = {'O': 0, 'Si': 1, 'Au': 2} # CHECK THIS

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    colors = sns.color_palette('deep')

    allatoms = read(path, index=':')

    pos = np.array([atoms.get_positions().flatten() for atoms in allatoms])
    cell = np.array([atoms.cell.array.flatten() for atoms in allatoms])
    types = [type_map[a.symbol] for a in allatoms[0]]

    models = [DeepPot(g) for g in graphs]

    dev = calc_model_devi(pos, cell, types, models, nopbc=False)[:, 4] # appends for each iterations


    sns.kdeplot(dev, color=colors[0], fill=True)

    # "ghost" lines for legend
    ax.plot([], [], lw=2.0, color=colors[0])

    ax.set_xlabel('$\epsilon$$_t$', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_xlim([0.00, 0.40])
    plt.tick_params(labelsize=12)
    plt.legend()
    plt.title(name)
    plt.savefig(name + ".pdf")
    file.close()
    
def make_std_hist_new(name = None, paths = None, graphs = None, type_map = None, devs = None):
    # i think what I'm going to do is run this in its own folder in the parent folder
    file = open("num_good_total.txt","w")
    type_map = type_map #{'O': 0, 'Si': 1, 'Au': 2} # CHECK THIS

    devs = devs
    for i,dev in enumerate(devs):
        for j,val in enumerate(dev):
            if val <= 0.1:
                good.append(j)

        file.write(path + " " +  str(len(good)))
        file.write("\n")

        sns.kdeplot(dev, color=colors[i], fill=True)

        # "ghost" lines for legend
        label = "Iteration_" + str(i+1)
        ax.plot([], [], lw=2.0, color=colors[i], label=label)

    ax.set_xlabel('$\epsilon$$_t$', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_xlim([0.00, 0.40])
    plt.tick_params(labelsize=12)
    plt.legend()
    plt.savefig(name + ".pdf")
    file.close()
    
def check_num_good(names = None,paths = None, graphs = None, hist_folder = None, final_pdf = None):
    file = open("num_good_total.txt_10","w")
    working_dir = os.getcwd()
    #pdfs = []

    type_map = {'O': 0, 'Si': 1, 'Au': 2} # CHECK THIS

    for i,path in enumerate(paths):
        good = []
        allatoms = read(path, index=':')

        pos = np.array([atoms.get_positions().flatten() for atoms in allatoms])
        cell = np.array([atoms.cell.array.flatten() for atoms in allatoms])
        types = [type_map[a.symbol] for a in allatoms[0]]

        graphs = graphs

        models = [DeepPot(g) for g in graphs]

        #dev = calc_model_devi(pos, cell, types, models, nopbc=False)

        dev = calc_model_devi(pos, cell, types, models, nopbc=False)[:, 4]

        for j,val in enumerate(dev):
            if val <= 0.1:
                good.append(j)

        file.write(path + " " +  str(len(good)))
        file.write("\n")

    file.close()
    
def density_scatter(x, y, name = None, fig=None, ax=None, sort=True, bins=300, **kwargs):
    """
    x = DFT data
    y = DP data"""
    if ax is None :
        fig , ax = plt.subplots()
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method="splinef2d", bounds_error=False)

    z[np.where(np.isnan(z))] = 0.0

    if sort :
        idx = z.argsort()

        x, y, z = x[idx], y[idx], z[idx]
        
    plt.plot(x,x,"r--")
    ax.scatter(x, y, s=0.1, c=z, cmap='Spectral_r', **kwargs)
    ax.set_xlabel("DFT Forces [eV/"r"$\AA$]")
    ax.set_ylabel("DP Forces [eV/"r"$\AA$]")
    plt.tight_layout()
    plt.savefig(name + ".pdf")


    
def make_std_line(name = None,path = None,gs = None):
    # i think what I'm going to do is run this in its own folder in the parent folder
    type_map = {'O': 0, 'Si': 1, 'Au': 2} # CHECK THIS

    devs = []

    allatoms_images = read(path, index=':')
    step = range(len(allatoms_images))
    step = list(step)
    for val in step:
        val *= 10 # puts the units into fs

    for allatoms in allatoms_images:
        pos = np.array([allatoms.get_positions().flatten()])
        cell = np.array([allatoms.cell.array.flatten()])
        types = [type_map[a.symbol] for a in allatoms]

        models = [DeepPot(g) for g in gs]

        devs.append(calc_model_devi(pos, cell, types, models, nopbc=False)[:, 4])

    fig = plt.figure(figsize = (20,10))
    plt.plot(step, devs, "b")
    plt.xlabel("Time Step [fs]")
    plt.ylabel("Standard Deviation")
    plt.savefig(name + ".pdf")
    plt.close(fig)
    
"""
def DFTMDvDP():
DFT_Fx_Au = []
DP_Fx_Au = []
DFT_Fy_Au = []
DP_Fy_Au = []
DFT_Fz_Au = []
DP_Fz_Au = []

DFT_Fx_Si = []
DP_Fx_Si = []
DFT_Fy_Si = []
DP_Fy_Si = []
DFT_Fz_Si = []
DP_Fz_Si = []

DFT_Fx_O = []
DP_Fx_O = []
DFT_Fy_O = []
DP_Fy_O = []
DFT_Fz_O = []
DP_Fz_O = []

step = []
DFT = read("md_PBE_400_111/vasprun.xml", index = ":")

for i in range(100,len(DFT)):
    step.append(0.5*(i-100))
    
    DFT_Fx_Au.append(DFT[i].get_forces()[:,0][-1])
    DFT_Fy_Au.append(DFT[i].get_forces()[:,1][-1])
    DFT_Fz_Au.append(DFT[i].get_forces()[:,2][-1])

    DFT_Fx_O.append(DFT[i].get_forces()[:,0][71])
    DFT_Fy_O.append(DFT[i].get_forces()[:,1][71])
    DFT_Fz_O.append(DFT[i].get_forces()[:,2][71])

    DFT_Fx_Si.append(DFT[i].get_forces()[:,0][71+36])
    DFT_Fy_Si.append(DFT[i].get_forces()[:,1][71+36])
    DFT_Fz_Si.append(DFT[i].get_forces()[:,2][71+36])

    del DFT[i].calc
    DFT[i].calc = DP(model = "graph.pb")
    DP_Fx_Au.append(DFT[i].get_forces()[:,0][-1])
    DP_Fy_Au.append(DFT[i].get_forces()[:,1][-1])
    DP_Fz_Au.append(DFT[i].get_forces()[:,2][-1])

    DP_Fx_O.append(DFT[i].get_forces()[:,0][71])
    DP_Fy_O.append(DFT[i].get_forces()[:,1][71])
    DP_Fz_O.append(DFT[i].get_forces()[:,2][71])
    
    DP_Fx_Si.append(DFT[i].get_forces()[:,0][71+36])
    DP_Fy_Si.append(DFT[i].get_forces()[:,1][71+36])
    DP_Fz_Si.append(DFT[i].get_forces()[:,2][71+36])

plt.plot(step,DP_Fx_Au,"r--",alpha = 0.7, label = "DPD")
plt.plot(step,DFT_Fx_Au,"r", linewidth = 3, label = "DFT Fx")
plt.plot(step,DFT_Fy_Au,"b", linewidth = 3, label = "DFT Fy")
plt.plot(step,DP_Fy_Au,"b--",alpha = 0.7)
plt.plot(step,DFT_Fz_Au,"g", linewidth = 3, label = "DFT Fz")
plt.plot(step,DP_Fz_Au,"g--",alpha = 0.7,)
plt.xlabel("Time step [fs]")
plt.ylabel("Force [eV/A]")
plt.legend()
plt.savefig("DFTvDP_F_Au.pdf")
plt.close()

plt.plot(step,DP_Fx_O,"r--",alpha = 0.7, label = "DPD")
plt.plot(step,DFT_Fx_O,"r", linewidth = 3, label = "DFT Fx")
plt.plot(step,DFT_Fy_O,"b", linewidth = 3, label = "DFT Fy")
plt.plot(step,DP_Fy_O,"b--",alpha = 0.7)
plt.plot(step,DFT_Fz_O,"g", linewidth = 3, label = "DFT Fz")
plt.plot(step,DP_Fz_O,"g--",alpha = 0.7,)
plt.xlabel("Time step [fs]")
plt.ylabel("Force [eV/A]")
plt.legend()
plt.savefig("DFTvDP_F_O.pdf")
plt.close()

plt.plot(step,DP_Fx_Si,"r--",alpha = 0.7, label = "DPD")
plt.plot(step,DFT_Fx_Si,"r", linewidth = 3, label = "DFT Fx")
plt.plot(step,DFT_Fy_Si,"b", linewidth = 3, label = "DFT Fy")
plt.plot(step,DP_Fy_Si,"b--",alpha = 0.7)
plt.plot(step,DFT_Fz_Si,"g", linewidth = 3, label = "DFT Fz")
plt.plot(step,DP_Fz_Si,"g--",alpha = 0.7,)
plt.xlabel("Time step [fs]")
plt.ylabel("Force [eV/A]")
plt.legend()
plt.savefig("DFTvDP_F_Si.pdf")
plt.close()

DFT_energy = []
DP_energy = []
step = []
DFT = read("md_PBE_400_111/vasprun.xml", index = ":")

for i in range(100,len(DFT)):
    step.append(0.5*(i-100))

    DFT_energy.append(DFT[i].get_potential_energy())

    del DFT[i].calc
    DFT[i].calc = DP(model = "graph.pb")
    DP_energy.append(DFT[i].get_potential_energy())

plt.plot(step,DFT_energy,"r", linewidth = 5, label = "DFT")
plt.plot(step,DP_energy,"k--",alpha = 0.7, label = "DPD")
plt.xlabel("Time step [fs]")
plt.ylabel("Energy eV")
plt.legend()
plt.savefig("DFTvDPMD_E.pdf")
plt.close()
"""

def find_train_configs(path = None, graphs = None):
    type_map = {'O': 0, 'Si': 1, 'Au': 2} # CHECK THIS

    allatoms = read(path+'/DPMD.traj', index=':')

    pos = np.array([atoms.get_positions().flatten() for atoms in allatoms])
    cell = np.array([atoms.cell.array.flatten() for atoms in allatoms])
    types = [type_map[a.symbol] for a in allatoms[0]]

    graphs = graphs

    models = [DP(g) for g in graphs]

    dev = calc_model_devi(pos, cell, types, models, nopbc=False)[:, 4]

    # deviation bounds
    lo = 0.10
    hi = 0.25

    # indices of images with lo < dev < hi
    new_configs = [i for i in range(len(dev)) if dev[i] > lo and dev[i] < hi]

    # take 500 new images (or however many there are if < 300)
    n_sample = 500 if len(new_configs) > 500 else len(new_configs)
    new_configs = random.sample(new_configs, n_sample) # randomly select those 500

    new_atoms = [allatoms[i] for i in new_configs]
    write(path+'/new_train_configs.traj', new_atoms)
    
""" 
def calc_model_devi_atom_specific(coord,
                    box,
                    atype,
                    models,
                    atoms,
                    fname=None,
                    frequency=1,
                    ):
    #if box is not None:
        #nopbc = True
    #else:
        #nopbc = False
    nopbc = False

    num_Au = 0
    len_atoms = len(atoms)
    for atom in atoms:
        if atom.symbol == "Au":
            num_Au +=1

    forces = []
    virials = []
    for dp in models:
        ret = dp.eval(
            coord,
            box,
            atype,
        )
        forces.append(ret[1])
        virials.append(ret[2] / len(atype))

    forces = np.array(forces)[:][:][:-num_Au]
    #print()
    #print("len of forces = ", len(forces[0][0][0])) # structure [number of graphs][number of images in the traj][number of atoms][force components]
    
virials = np.array(virials)[:][:][:-num_Au]

    devi = [np.arange(coord.shape[0]) * frequency]
    devi += list(calc_model_devi_v(virials))
    devi += list(calc_model_devi_f(forces))
    devi = np.vstack(devi).T
    if fname:
        write_model_devi_out(devi, fname)
    return devi

def make_std_hist_atom_specific(name = None, path = None, graphs = None):
    # i think what I'm going to do is run this in its own folder in the parent folder
    type_map = {'O': 0, 'Si': 1, 'Au': 2} # CHECK THIS

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    colors = sns.color_palette('deep')
    allatoms = read(path, index=':')

    pos = np.array([atoms.get_positions().flatten() for atoms in allatoms])
    cell = np.array([atoms.cell.array.flatten() for atoms in allatoms])
    types = [type_map[a.symbol] for a in allatoms[0]]

    gs = graphs

    models = [DeepPot(g) for g in gs]

    dev = calc_model_devi_atom_specific(pos, cell, types, models, atoms = allatoms[0])[:, 4] # appends for each iterations
    ns.kdeplot(dev, color=colors[0], fill=True)

    # "ghost" lines for legend
    ax.plot([], [], lw=2.0, color=colors[0])

    ax.set_xlabel('$\epsilon$$_t$', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_xlim([0.00, 0.40])
    plt.tick_params(labelsize=12)
    plt.legend()
    plt.savefig(name + ".pdf")
"""