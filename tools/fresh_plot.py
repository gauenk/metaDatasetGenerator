import matplotlib,glob,cv2
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fresh_config import cfg,create_model_name
from fresh_util import transpose_measures

def plot_template_function(xdata,ydata,line_type,xlabel,ylabel,title):
    hdle, = plt.plot(xdata,ydata,line_type)
    if title: plt.title(title)
    if ylabel: plt.ylabel(ylabel)
    if xlabel: plt.xlabel(xlabel)
    return hdle

def plot_add_legend(plot_handles,plot_handle_names):
    plt.legend(plot_handles, plot_handle_names)
    
def plot_clear():
    plt.clf(),plt.cla()

def plot_save_figure(filename):
    plt.savefig(filename)
    
def plot_measure_vs_clusters(train_scores,test_scores,kmeans_search_list,ylabel,title,filename):
    ave_scores = average_list_elements(train_scores,test_scores)
    train_handle = plot_template_function(kmeans_search_list,train_scores,'b+',
                                          '# of clusters',ylabel,title)
    test_handle = plot_template_function(kmeans_search_list,test_scores,'g^',
                                         None,None,None)
    ave_handle = plot_template_function(kmeans_search_list,ave_scores,'r*',
                                         None,None,None)
    plot_add_legend([train_handle,test_handle,ave_handle],['train','test','ave'])
    plot_save_figure(filename)
    plot_clear()

def plot_aggregate_scores_vs_clusters(train_scores,test_scores,kmeans_search_list):
    plot_measure_vs_clusters(train_scores,test_scores,kmeans_search_list,'Combined [via sum] Score [Measure]',
                             'Choosing Optimal K [{}]'.format(cfg.modelInfo.name),
                             'measure_combined_vs_cluster_model_{}.png'.format(cfg.modelInfo.name))

def plot_measure(train_measure,test_measure,measure_name,kmeans_search_list):
    plot_measure_vs_clusters(train_measure,test_measure,kmeans_search_list,'{} [Measure] '.format(measure_name),
                             'Choosing Optimal K [{}]'.format(cfg.modelInfo.name),
                             'measure_{}_vs_cluster_model_{}.png'.format(measure_name,cfg.modelInfo.name))

def plot_measure_list(train_measure_list,test_measure_list,kmeans_search_list):
    """
    measure_list:
       -> len(measure_list) = # of k's in search list
       -> len(measure_list[index]) = # of measures
    measure_list_transpose:
       -> len(measure_list) = # of measures
       -> len(measure_list[index]) = # of k's in search list
    """
    measure_name_dict = {'cluster':['silhouette','homogeneity_ds_labels','homogeneity_correct'],
                         'separability':['separability_ds_labels','separability_ds_correct']}
    for measure_type in train_measure_list.keys():
        measure_name_list = measure_name_dict[measure_type]
        train_measure_type_list_t,measure_name_list_t = transpose_measures(train_measure_list[measure_type],measure_name_list)
        test_measure_type_list_t,measure_name_list_t = transpose_measures(test_measure_list[measure_type],measure_name_list)
        for train_measure,test_measure,measure_name in zip(train_measure_type_list_t,test_measure_type_list_t,measure_name_list_t):
            print(measure_name)
            plot_measure(train_measure,test_measure,measure_name,kmeans_search_list)


def init_plot(plot_dict,fieldname,msr_name):
    title = fieldname
    # title = fieldname + '_' + msr_name
    plot_dict[msr_name] = plt.subplots()
    plot_dict[msr_name][1].set_title(title)
    plot_dict[msr_name][1].set_xlabel('number of clusters')
    plot_dict[msr_name][1].set_ylabel(msr_name)

def add_plot_data(plot_dict,msr_name,msr_value,fieldname,fieldvalue,setid,marker,kmeans_search_list):
    # label = setid + '_' + fieldname + '_' + str(fieldvalue)
    label = str(fieldvalue)
    if setid == 'train': label = '_'+label
    if setid == 'train': fmt = 'b'+marker+'-'
    elif setid == 'test': fmt = 'g'+marker+'-'
    plot_dict[msr_name][1].plot(kmeans_search_list,msr_value,fmt,label=label)


def average_list_elements(list_a,list_b):
    list_c = [(a+b)/2. for a,b in zip(list_a,list_b)]
    return list_c
        

def tile_by_comboid(combolist):
    xsep = 10 # number of pixels between images along x
    ysep = 10 # number of pixels between images along y
    
    for comboid in combolist:
        imglist = []
        for filename in glob.glob("measure*+{}_*png".format(comboid)):
            if 'tile' in filename: continue # don't add to itself
            imglist.append(cv2.imread(filename))
        yshapes = [image.shape[0] for image in imglist]
        xshapes = [image.shape[1] for image in imglist]
        shape = [max(yshapes),sum(xshapes)+xsep*len(xshapes),3]
        tile = np.zeros(shape)
        x_index_start,x_index_end = 0,0
        for index,image in enumerate(imglist):
            if index != 0: x_index_start += xshapes[index-1] + xsep
            x_index_end = x_index_start + xshapes[index]
            tile[:image.shape[0],x_index_start:x_index_end,:] = image
        cv2.imwrite("tile_{}_{}.png".format(comboid,cfg.modelInfo.name),tile)
                                                  

def mangle_plot_name(modelInfo,fieldname):
    import copy
    modelInfo_copy = copy.deepcopy(modelInfo)
    modelInfo_copy[fieldname] = '[{}]'.format(fieldname)
    return create_model_name(modelInfo_copy)
    

def analyze_expfield_results(train_results,test_results,exp_configs,expfield_filter,kmeans_search_list):
    measure_name_dict = {'cluster':['silhouette','homogeneity_ds_labels','homogeneity_correct'],
                         'separability':['separability_ds_labels','separability_ds_correct']}
    change_field = find_experiment_changes(exp_configs)    
    fieldname_list = [field[0] for field in change_field]
    unique_changes = get_unique_strings(fieldname_list)
    marker_list = cfg.plot.marker_list
    marker_index = dict.fromkeys(unique_changes,0)
    print(marker_index)
    plot_dict = {}
    handle_dict = {}
    for field_index,exp_config in enumerate(exp_configs):
        # update to exp_config
        old_cfg = deepcopy(cfg)
        update_config(cfg,exp_config)
        fieldname,fieldvalue = what_changed(cfg,old_cfg)
        if fieldname != expfield_filter: continue
        reset_model_name()

        # update to exp_config
        train_msrs,test_msrs = train_results[field_index],test_results[field_index]
        print(fieldname)
        print(train_msrs)
        # if fieldname not in handle_dict.keys(): handle_dict[fieldname] = {}
        for measure_type in train_msrs.keys():
            msr_dict_type = measure_name_dict[measure_type]
            print("----------------------------------")
            print(msr_dict_type)
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print(train_msrs[measure_type])
            print(test_msrs[measure_type])
            tr_type_t,tr_msr_name_t = transpose_measures(train_msrs[measure_type],msr_dict_type)
            te_type_t,te_msr_name_t = transpose_measures(test_msrs[measure_type],msr_dict_type)
            print("************************************")
            # always in the same order
            for tr_msr,te_msr in zip(tr_msr_name_t,te_msr_name_t):
                assert tr_msr == te_msr,"train/test is not the same order"
            print("====================================")
            for tr_msr,te_msr,msr_name in zip(tr_type_t,te_type_t,tr_msr_name_t):
                plot_title = mangle_plot_name(cfg.modelInfo,fieldname)
                # if msr_name not in handle_dict[fieldname].keys(): handle_dict[fieldname][msr_name] = []
                if msr_name not in plot_dict.keys(): init_plot(plot_dict,plot_title,msr_name)
                add_plot_data(plot_dict,msr_name,tr_msr,fieldname,fieldvalue,'train',marker_list[marker_index[fieldname]],kmeans_search_list)
                add_plot_data(plot_dict,msr_name,te_msr,fieldname,fieldvalue,'test',marker_list[marker_index[fieldname]],kmeans_search_list)
        marker_index[fieldname] += 1

    for exp_change in unique_changes:
    # for exp_config in exp_configs:
        # update to exp_config
        # old_cfg = deepcopy(cfg)
        # update_config(cfg,exp_config)
        # fieldname,fieldvalue = what_changed(cfg,old_cfg)
        # if fieldname != expfield_filter: continue
        # reset_model_name()
        for measure_type in train_msrs.keys():
            msr_dict_type = measure_name_dict[measure_type]
            _,tr_msr_name_t = transpose_measures(train_msrs[measure_type],msr_dict_type)
            _,te_msr_name_t = transpose_measures(test_msrs[measure_type],msr_dict_type)
            for tr_msr,te_msr in zip(tr_msr_name_t,te_msr_name_t):
                assert tr_msr == te_msr,"train/test is not the same order"
            for msr_name in tr_msr_name_t:
                if msr_name not in plot_dict.keys(): init_plot(plot_dict,msr_name)
                save_name = msr_name+'_'+ exp_change + '.png'
                ylim = plot_dict[msr_name][1].get_ylim()
                xlim = plot_dict[msr_name][1].get_xlim()
                ylim = [ylim[0]-.01*ylim[1],1.01*ylim[1]]
                xlim = [xlim[0]-.05*xlim[1],1.05*xlim[1]]
                # train_handle, = plot_dict[msr_name][1].plot(None,None,'b')
                # test_handle, = plot_dict[msr_name][1].plot(None,None,'r')
                # train_test_legend = plt.legend([train_handle,test_handle],['train','test'])
                # lines = [matplotlib.lines.Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in ['b','r']]
                # train_test_legend = plt.legend(lines,['train','test'])
                # plot_dict[msr_name][1].add_artist(train_test_legend)
                train_patch = mpatches.Patch(color='blue',label='train')
                test_patch = mpatches.Patch(color='green',label='test')
                plot_dict[msr_name][1].set_ylim(ylim)
                plot_dict[msr_name][1].set_xlim(xlim)
                plot_dict[msr_name][0].subplots_adjust(right=0.7)
                plot_dict[msr_name][0].tight_layout(rect=[0,0,0.75,1])
                plot_handles = [train_patch,test_patch]
                plot_labels = ['train','test']
                handles,labels = plot_dict[msr_name][1].get_legend_handles_labels()
                plot_handles += handles
                plot_labels += labels
                plot_dict[msr_name][1].legend(handles=plot_handles,labels=plot_labels,loc='center left', bbox_to_anchor=(1, 0.5), title=exp_change)
                plot_dict[msr_name][0].savefig(save_name,bbox_inches='tight')

def genErrorVsCorrIncorrSep(train_results,test_results,exp_configs,expfield_filter,kmeans_search_list):
    measure_name_dict = cfg.measure_name_dict
    change_field = find_experiment_changes(exp_configs)    
    fieldname_list = [field[0] for field in change_field]
    unique_changes = get_unique_strings(fieldname_list)
    marker_list = ['o','v','^','<','>','8','s','p','h','H','+','x','X','D','d']
    marker_index = dict.fromkeys(unique_changes,0)
    print(marker_index)
    plot_dict = {}
    handle_dict = {}
    for field_index,exp_config in enumerate(exp_configs):
        # update to exp_config
        old_cfg = deepcopy(cfg)
        update_config(cfg,exp_config)
        fieldname,fieldvalue = what_changed(cfg,old_cfg)
        if fieldname != expfield_filter: continue
        reset_model_name()

        # update to exp_config
        train_msrs,test_msrs = train_results[field_index],test_results[field_index]
        print(fieldname)
        print(train_msrs)
        # if fieldname not in handle_dict.keys(): handle_dict[fieldname] = {}
        for measure_type in train_msrs.keys():
            msr_dict_type = measure_name_dict[measure_type]
            print("----------------------------------")
            print(msr_dict_type)
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print(train_msrs[measure_type])
            print(test_msrs[measure_type])
            tr_type_t,tr_msr_name_t = transpose_measures(train_msrs[measure_type],msr_dict_type)
            te_type_t,te_msr_name_t = transpose_measures(test_msrs[measure_type],msr_dict_type)
            print("************************************")
            # always in the same order
            for tr_msr,te_msr in zip(tr_msr_name_t,te_msr_name_t):
                assert tr_msr == te_msr,"train/test is not the same order"
            print("====================================")
            for tr_msr,te_msr,msr_name in zip(tr_type_t,te_type_t,tr_msr_name_t):
                plot_title = mangle_plot_name(cfg.modelInfo,fieldname)
                # if msr_name not in handle_dict[fieldname].keys(): handle_dict[fieldname][msr_name] = []
                if msr_name not in plot_dict.keys(): init_plot(plot_dict,plot_title,msr_name)
                add_plot_data(plot_dict,msr_name,tr_msr,fieldname,fieldvalue,'train',marker_list[marker_index[fieldname]],kmeans_search_list)
                add_plot_data(plot_dict,msr_name,te_msr,fieldname,fieldvalue,'test',marker_list[marker_index[fieldname]],kmeans_search_list)
        marker_index[fieldname] += 1

    for exp_change in unique_changes:
    # for exp_config in exp_configs:
        # update to exp_config
        # old_cfg = deepcopy(cfg)
        # update_config(cfg,exp_config)
        # fieldname,fieldvalue = what_changed(cfg,old_cfg)
        # if fieldname != expfield_filter: continue
        # reset_model_name()
        for measure_type in train_msrs.keys():
            msr_dict_type = measure_name_dict[measure_type]
            _,tr_msr_name_t = transpose_measures(train_msrs[measure_type],msr_dict_type)
            _,te_msr_name_t = transpose_measures(test_msrs[measure_type],msr_dict_type)
            for tr_msr,te_msr in zip(tr_msr_name_t,te_msr_name_t):
                assert tr_msr == te_msr,"train/test is not the same order"
            for msr_name in tr_msr_name_t:
                if msr_name not in plot_dict.keys(): init_plot(plot_dict,msr_name)
                save_name = msr_name+'_'+ exp_change + '.png'
                ylim = plot_dict[msr_name][1].get_ylim()
                xlim = plot_dict[msr_name][1].get_xlim()
                ylim = [ylim[0]-.01*ylim[1],1.01*ylim[1]]
                xlim = [xlim[0]-.05*xlim[1],1.05*xlim[1]]
                # train_handle, = plot_dict[msr_name][1].plot(None,None,'b')
                # test_handle, = plot_dict[msr_name][1].plot(None,None,'r')
                # train_test_legend = plt.legend([train_handle,test_handle],['train','test'])
                # lines = [matplotlib.lines.Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in ['b','r']]
                # train_test_legend = plt.legend(lines,['train','test'])
                # plot_dict[msr_name][1].add_artist(train_test_legend)
                # handles,labels = plot_dict[msr_name][1].get_legend_handles_labels()
                # plot_handles += handles
                # plot_labels += labels
                plot_handles,plot_labels = add_train_test_patch_handles(plot_dict[msr_name][1])
                plot_dict[msr_name][1].set_ylim(ylim)
                plot_dict[msr_name][1].set_xlim(xlim)
                plot_dict[msr_name][0].subplots_adjust(right=0.7)
                plot_dict[msr_name][0].tight_layout(rect=[0,0,0.75,1])
                plot_dict[msr_name][1].legend(handles=plot_handles,labels=plot_labels,loc='center left', bbox_to_anchor=(1, 0.5), title=exp_change)
                plot_dict[msr_name][0].savefig(save_name,bbox_inches='tight')

    
def add_train_test_patch_handles(axis_var):
    train_patch = mpatches.Patch(color=cfg.plot.color_dict['train'],label='train')
    test_patch = mpatches.Patch(color=cfg.plot.color_dict['test'],label='test')
    plot_handles = [train_patch,test_patch]
    plot_labels = ['train','test']
    handles,labels = axis_var.get_legend_handles_labels()
    plot_handles += handles
    plot_labels += labels
    return plot_handles,plot_labels
