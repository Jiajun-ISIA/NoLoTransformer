import faiss, matplotlib.pyplot as plt, os, numpy as np, torch
from PIL import Image
import pdb


#######################
def evaluate(dataset, LOG, metric_computer, dataloaders, model, opt, evaltypes, device,
             aux_store=None, make_recall_plot=False, store_checkpoints=True, log_key='Test'):
    """
    Parent-Function to compute evaluation metrics, print summary string and store checkpoint files/plot sample recall plots.
    """
    computed_metrics, extra_infos = metric_computer.compute_standard(opt, model, dataloaders[0], evaltypes, device)

    numeric_metrics = {}
    histogr_metrics = {}
    for main_key in computed_metrics.keys():
        for name,value in computed_metrics[main_key].items():
            if isinstance(value, np.ndarray):
                if main_key not in histogr_metrics: histogr_metrics[main_key] = {}
                histogr_metrics[main_key][name] = value
            else:
                if main_key not in numeric_metrics: numeric_metrics[main_key] = {}
                numeric_metrics[main_key][name] = value

    ###
    full_result_str = ''
    for evaltype in numeric_metrics.keys():
        full_result_str += 'Embed-Type: {}:\n'.format(evaltype)
        for i,(metricname, metricval) in enumerate(numeric_metrics[evaltype].items()):
            full_result_str += '{0}{1}: {2:4.4f}'.format(' | ' if i>0 else '',metricname, metricval)
        full_result_str += '\n'

    print(full_result_str)


    ###
    for evaltype in evaltypes:
        for storage_metric in opt.storage_metrics:
            parent_metric = evaltype+'_{}'.format(storage_metric.split('@')[0])
            if parent_metric not in LOG.progress_saver[log_key].groups.keys() or \
               numeric_metrics[evaltype][storage_metric]>np.max(LOG.progress_saver[log_key].groups[parent_metric][storage_metric]['content']):
               print('Saved weights for best {}: {}\n'.format(log_key, parent_metric))
               set_checkpoint(model, opt, LOG.progress_saver, LOG.prop.save_path+'/checkpoint_{}_{}_{}.pth.tar'.format(log_key, evaltype, storage_metric), aux=aux_store)


    ###
    if opt.log_online:
        for evaltype in histogr_metrics.keys():
            for eval_metric, hist in histogr_metrics[evaltype].items():
                import wandb, numpy
                wandb.log({log_key+': '+evaltype+'_{}'.format(eval_metric): wandb.Histogram(np_histogram=(list(hist),list(np.arange(len(hist)+1))))}, step=opt.epoch)
                wandb.log({log_key+': '+evaltype+'_LOG-{}'.format(eval_metric): wandb.Histogram(np_histogram=(list(np.log(hist)+20),list(np.arange(len(hist)+1))))}, step=opt.epoch)

    ###
    for evaltype in numeric_metrics.keys():
        for eval_metric in numeric_metrics[evaltype].keys():
            parent_metric = evaltype+'_{}'.format(eval_metric.split('@')[0])
            LOG.progress_saver[log_key].log(eval_metric, numeric_metrics[evaltype][eval_metric],  group=parent_metric)

        ###
        if make_recall_plot:
            recover_closest_standard(extra_infos[evaltype]['features'],
                                     extra_infos[evaltype]['image_paths'],
                                     extra_infos[evaltype]['target_labels'],
                                     LOG.prop.save_path+'/sample_recoveries.png')


###########################
def set_checkpoint(model, opt, progress_saver, savepath, aux=None):
    if 'experiment' in vars(opt):
        import argparse
        save_opt = {key:item for key,item in vars(opt).items() if key!='experiment'}
        save_opt = argparse.Namespace(**save_opt)
    else:
        save_opt = opt

    torch.save({'state_dict':model.state_dict(), 'opt':save_opt, 'progress':progress_saver, 'aux':aux}, savepath)




##########################
def recover_closest_standard(feature_matrix_all, image_paths, labels, save_path, n_image_samples=20, n_closest=10):
    
    image_paths = np.array([x[0] for x in image_paths])
    
    feature_matrix_all = feature_matrix_all.cpu().numpy()
    faiss_search_index = faiss.IndexFlatL2(feature_matrix_all.shape[-1])
    faiss_search_index.add(feature_matrix_all)
    _, closest_feature_idxs = faiss_search_index.search(feature_matrix_all, n_closest+1)

    # pdb.set_trace()
    for k in range(1):
        sample_idxs = np.random.choice(np.arange(len(feature_matrix_all)), n_image_samples)
        sample_paths = image_paths[closest_feature_idxs][sample_idxs]
        sample_labels = labels[closest_feature_idxs][sample_idxs]
        # sample_paths = image_paths[closest_feature_idxs][]
        # sample_labels = labels[closest_feature_idxs][sample_idxs]
        for i in range(sample_labels.shape[0]):
            for j in range(1,sample_labels.shape[1]):
                sample_labels[i][j] = int(sample_labels[i][j] == sample_labels[i][0])
        sample_labels = sample_labels.reshape(-1)
    
        # pdb.set_trace()
        f,axes = plt.subplots(n_image_samples, n_closest+1)
        for i,(ax,plot_path) in enumerate(zip(axes.reshape(-1), sample_paths.reshape(-1))):
            img = Image.open(plot_path)
            img = img.resize((400, 255),Image.ANTIALIAS)
            # img.resize(400,255)
            ax.imshow(np.array(img))
            ax.set_xticks([])
            ax.set_yticks([])
            if i%(n_closest+1):
                if sample_labels[i]:
                    ax.axvline(x=0, color='g', linewidth=13)
                else:
                    ax.axvline(x=0, color='r', linewidth=13)
            else:
                ax.axvline(x=0, color='orange', linewidth=13)
        f.set_size_inches(40,80)
        f.tight_layout()
        save_path = save_path.split('/')
        save_path[-1] = str(k) + save_path[-1]
        save_path = '/'.join(save_path)
        f.savefig(save_path, dpi=100)
        plt.close()
