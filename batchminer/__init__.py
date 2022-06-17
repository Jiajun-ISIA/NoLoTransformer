from batchminer import softhard,random,distance

BATCHMINING_METHODS = {'random':random,
                       'softhard':softhard,
                       'distance':distance,}
  


def select(batchminername, opt):
    #####
    if batchminername not in BATCHMINING_METHODS: raise NotImplementedError('Batchmining {} not available!'.format(batchminername))

    batchmine_lib = BATCHMINING_METHODS[batchminername]

    return batchmine_lib.BatchMiner(opt)
