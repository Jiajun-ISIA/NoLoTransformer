import architectures.ours_model



def select(arch, opt):
    if 'ours_model' in arch:
        return ours_model.Network(opt)
