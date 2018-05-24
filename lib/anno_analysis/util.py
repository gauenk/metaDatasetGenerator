import re,sys
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio

def plot_people_mask(people_mask,dataset_name,rescaled=True,log=False):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    if log:
        people_mask = np.ma.log(people_mask)/np.ma.log(people_mask).sum()
    my_max = np.max(people_mask)
    my_min = np.min(people_mask)
    my_mean = np.mean(people_mask)

    print("min: {0:4.4f} \nmax: {1:4.4f}\nmean: {2:4.4f}".format(my_min,my_max,my_mean))

    if rescaled:
        cax = ax.imshow(people_mask,vmin=0,cmap="coolwarm",interpolation="none")
        fn = '{}_people_density_rescaled.png'.format(dataset_name)
    else:
        cax = ax.imshow(people_mask,vmin=0,vmax=0.5,cmap="coolwarm",interpolation="none")
        fn = '{}_people_density.png'.format(dataset_name)

    # bar in [0,1]
    cbar = fig.colorbar(cax,ticks=[my_min,my_mean,my_max])
    cbar.ax.set_yticklabels(['< {0:2.2f}%'.format(my_min*100), '< {0:2.2f}%'.format(my_mean*100), '> {0:2.2f}%'.format(my_max*100)],size=50)
    # cbar = fig.colorbar(cax,ticks=[my_max])
    # cbar.ax.set_yticklabels(['> {0:2.2f}%'.format(my_max*100)],size=30)

    # bar in [min_value,max_value]
    #cbar = fig.colorbar(cax,ticks=[my_min,my_max])
    #cbar.ax.set_yticklabels(['< {0:2.2f}%'.format(my_min*100), '> {0:2.2f}%'.format(my_max*100)])
    ax.axis("off")
    plt.savefig(fn,bbox_inches='tight')

    
## PLOT LOG PEOPLE_MASK ##
def plot_people_mask_log_norm(people_mask,dataset_name,rescaled=True):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    people_mask = np.ma.log(people_mask)/np.ma.log(people_mask).sum()
    print(people_mask)
    my_max = np.max(people_mask)
    my_min = np.min(people_mask)
    my_mean = np.mean(people_mask)
    print("min: {0:4.4f} \nmax: {1:4.4f}\nmean: {2:4.4f}".format(my_min,my_max,my_mean))
    log_mask = people_mask
    if rescaled:
        cax = ax.imshow(log_mask,vmin=0,cmap="coolwarm",interpolation="none")
        fn = '{}_people_density_rescaled.png'.format(dataset_name)
    else:
        cax = ax.imshow(people_mask,vmin=0,vmax=0.5,cmap="coolwarm",interpolation="none")
        fn = '{}_people_density.png'.format(dataset_name)
    #cbar = fig.colorbar(cax,ticks=np.ma.log([my_min,my_mean,my_max]))
    cbar = fig.colorbar(cax,ticks=[my_min,my_mean,my_max])
    cbar.ax.set_yticklabels(['< {0:2.2f}%'.format(my_min*100), '< {0:2.2f}%'.format(my_mean*100), '> {0:2.2f}%'.format(my_max*100)],size=50)
    ax.axis("off")
    plt.savefig(fn,bbox_inches='tight')


    
