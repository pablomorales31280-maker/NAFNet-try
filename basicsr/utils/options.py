# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import yaml
from collections import OrderedDict
from os import path as osp


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path, args, is_train=True):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        is_train (str): Indicate whether in training or not. Default: True.

    Returns:
        (dict): Options.
    """
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    if args.learn_alpha:
        args.use_alpha = True
    opt['is_train'] = is_train
    opt['flc'] = args.flc_pooling
    opt['use_alpha'] = args.use_alpha
    opt['learn_alpha'] = args.learn_alpha
    opt['adv_train'] = args.adv_train
    opt['attacking'] = args.attacking
    opt['use_GELU'] = args.gelu
    opt['use_blur'] = args.blur
    opt['kernel_size'] = args.kernel_size
    opt['show_alpha'] = args.show_alpha
    opt['para_kernel_size'] = args.para_kernel_size
    if args.flc_pooling:
        opt['network_g']['type'] += 'FLC'
    if args.gelu2:
        opt['network_g']['type'] += 'GELU2'
    if args.gelu:
        opt['network_g']['type'] += 'GELU'
    if args.relu:
        opt['network_g']['type'] += 'ReLU'
    if args.leaky_relu:
        opt['network_g']['type'] += 'LeakyReLU'
    #if args.use_alpha:
    opt['network_g']['use_alpha'] = args.use_alpha
    #if args.learn_alpha:
    opt['network_g']['learn_alpha'] = args.learn_alpha
    opt['network_g']['use_blur'] = args.blur
    opt['network_g']['kernel_size'] = args.kernel_size
    opt['network_g']['para_kernel_size'] = args.para_kernel_size
    
    opt['network_g']['padding'] = args.padding
        
    opt['network_g']['use_conv'] = args.use_conv
    opt['network_g']['first_drop_alpha'] = args.first_drop_alpha
    opt['network_g']['drop_alpha'] = args.drop_alpha
    opt['network_g']['test_wo_drop_alpha'] = args.test_wo_drop_alpha
    opt['network_g']['test_drop_alpha'] = args.test_drop_alpha
    opt['network_g']['half_precision'] = args.half_precision
    opt['network_g']['upsampling_method'] = args.upsampling_method

    
    if args.pretrain_network_g != None:
        opt['path']['pretrain_network_g'] = args.pretrain_network_g
    if args.resume_state != None:
        opt['path']['resume_state'] = args.resume_state

    
    if 'attack' in opt:
        if args.attack is not None:
            opt['attack']['method'] = args.attack
        if args.iterations != 0:
            opt['attack']['iterations'] = args.iterations
        #opt['attack']['alpha'] = 0.04 if opt['attack']['method'] == 'cospgd' else 0.01
        if args.epsilon != 0.03:
            opt['attack']['epsilon'] = args.epsilon
        if args.alpha != 0.01:
            opt['attack']['alpha'] = args.alpha
        opt['attack']['targeted'] = args.targeted
        attack_opt = opt['attack'] 
        attack = attack_opt['method']
        iterations = str(attack_opt['iterations'])
        alpha = str(attack_opt['alpha'])
        epsilon = str(attack_opt['epsilon'])
        targeted = str(attack_opt['targeted'])
    else:        
        attack = 'no_attack'
        iterations, alpha, epsilon, targeted = 'none', 'none', 'none', 'none'
        
    
    if 'name' in opt:
        if args.kernel_size > 1:
            opt['name'] += '_trans_up_kernel_{}'.format(args.kernel_size)
            if args.para_kernel_size > 1:
                opt['name'] += '_para_trans_{}'.format(args.para_kernel_size)
        if args.flc_pooling:
            opt['name'] += '_FLC'
            if args.use_alpha:
                opt['name'] += '_alpha'
                if args.learn_alpha:
                    opt['name'] += '_learned' 
                else:
                    opt['name'] += '_random'
                if args.blur:
                    opt['name'] += '_blurred'
        if args.adv_train:
            opt['name'] += '_ADV_T_{}_eps_{}_alpha_{}_itrs_{}'.format(opt['attack']['method'], opt['attack']['epsilon'], opt['attack']['alpha'], opt['attack']['iterations'])
        if args.gelu2:
            opt['name'] += '_GELU2'
        elif args.gelu:
            opt['name'] += '_GELU'
        if args.relu:
            opt['name'] += '_ReLU'
        name = opt['name']     
        try:
            task = opt['datasets']['test']['name']
        except Exception:
            task = opt['datasets']['val']['name']
    else:
        name = 'god_knows'
        task = 'ask_the_devil'
    
    

    # datasets
    if 'datasets' in opt:
        for phase, dataset in opt['datasets'].items():
            # for several datasets, e.g., test_1, test_2
            phase = phase.split('_')[0]
            dataset['phase'] = phase
            if 'scale' in opt:
                dataset['scale'] = opt['scale']
            if dataset.get('dataroot_gt') is not None:
                dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
            if dataset.get('dataroot_lq') is not None:
                dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key
                                  or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)
    opt['path']['root'] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments', '{}'.format(args.folder),
                                    opt['name']+'{}'.format('_flc_pooling_low_freq_{}_{}_alpha_{}_{}_blurring_{}_padding_upsampling_{}'.format(opt['flc'], 
                                                                                                                                               'concat', 'learned' if args.learn_alpha else 'random', 
                                                                                                                                               'with' if args.blur else 'without', args.padding, 
                                                                                                                                               args.upsampling_method)if args.use_alpha else ''))
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root,
                                                  'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root,
                                                'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = osp.join(opt['path']['root'], 'experiments', '{}'.format(args.folder), task, 
                                opt['name']+'{}'.format('_flc_pooling_low_freq_{}_{}_alpha_{}_{}_blurring_{}_padding_upsampling_{}'.format(opt['flc'], 
                                                                                                                                               'concat', 'learned' if args.learn_alpha else 'random', 
                                                                                                                                               'with' if args.blur else 'without', args.padding, 
                                                                                                                                               args.upsampling_method)if args.use_alpha else ''), 
                                attack, iterations, 'alpha_'+alpha, 'eps_'+epsilon, 'targeted_'+targeted)
        #results_root = osp.join(opt['path']['root'], 'testing', task, opt['name'], attack, iterations, 'alpha_'+alpha, 'eps_'+epsilon, 'targeted_'+targeted)
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg