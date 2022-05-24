import architectures
from copy import deepcopy
adacapnetfast = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 200},
                "max_iter":200,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":256,"depth":2}           
                }
adacapnetstandard = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":512,"depth":2}           
                }

adacapnetfastselu = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 200},
                "max_iter":200,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":256,"depth":2, "activation":"SELU", "initializer_params":{"gain_type":'linear'}}           
                }
adacapnetselu = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":512,"depth":2, "activation":"SELU", "initializer_params":{"gain_type":'linear'}}           
                }

adacapnetglu = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.GLULayers,
                "hidden_params" :  {"width":512,"depth":3}           
                }

adacapnetresblock = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.ResidualLayers,
                "hidden_params" :  {"width":512,"depth":2,"block_depth":2}           
                }
adacapnetbatchstandard = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 20},
                "max_iter":20,
                "epochs":True,
                "max_runtime":3600,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":512,"depth":2}           
                }
adacapnetbatchresblock = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 50},
                "max_iter":50,
                "epochs":True,
                "max_runtime":3600,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.ResidualLayers,
                "hidden_params" :  {"width":512,"depth":2,"block_depth":2}           
                }

#Not added to the benchmark for compute time reasons
adacapnetresglu = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.ResidualGLULayers,
                "hidden_params" :  {"width":512,"depth":2,"block_depth":2}           
                }
adacapnetresreglu = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.ResidualGLULayers,
                "hidden_params" :  {"width":512,"depth":2,"block_depth":2,"gate_activation":"ReLU", "gate_initializer_params":{"gain_type":'relu'}}           
                }
adacapnetreglu = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.GLULayers,
                "hidden_params" :  {"width":512,"depth":3,"gate_activation":"ReLU", "gate_initializer_params":{"gain_type":'relu'}}
                }
adacapnetwide = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-3, "total_steps" : 1000},
                "max_iter":1000,
                "learning_rate":1e-4, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":2048,"depth":2}           
                }
adacapnetresblockselu = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.ResidualLayers,#Does not really make sense with SeLU activation in theory
                "hidden_params" :  {"width":512,"depth":2,"block_depth":2, "activation":"SELU", "initializer_params":{"gain_type":'linear'}} 
                }
adacapnetresblockwide = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-3, "total_steps" : 1000},
                "max_iter":1000,
                "learning_rate":1e-4, #useless with OneCycleLR
                "hidden_nn" : architectures.ResidualLayers,
                "hidden_params" :  {"width":2048,"depth":2,"block_depth":2}           
                }
adacapnetdeep = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 200},
                "max_iter":200,
                "batch_size":256,
                "validation_fraction":False,
              "max_runtime":3600,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":256,"depth":6}           
                }
adacapnetseludeep = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 200},
                "max_iter":200,
                "batch_size":256,
                "validation_fraction":False,
                "max_runtime":3600,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":256,"depth":6, "activation":"SELU", "initializer_params":{"gain_type":'linear'}}           
                }
adacapnetresblockdeep = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 200},
                "max_iter":200,
                "batch_size":256,
                "validation_fraction":False,
                "max_runtime":3600,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.ResidualLayers,
                "hidden_params" :  {"width":256,"depth":3,"block_depth":3}           
                }
adacapnetresblockdeepselu = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 200},
                "max_iter":200,
                "batch_size":256,
                "validation_fraction":False,
                "max_runtime":3600,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.ResidualLayers,
                "hidden_params" :  {"width":256,"depth":3,"block_depth":3, "activation":"SELU", "initializer_params":{"gain_type":'linear'}}
                }

#regular nets
for architecture in ["fast","standard","resblock","batchstandard","batchresblock", "glu", "reglu", "resglu", "resreglu", "wide", "resblockwide", "selu", "fastselu","seludeep", "deep","resblockdeep","resblockdeepselu"]:
    locals()["regularnet"+architecture] = deepcopy(eval("adacapnet"+architecture))
    locals()["regularnet"+architecture]["hidden_params"].update({"dropout":0.2,"batch_norm":True})
for architecture in ["batchstandard","batchresblock"]:
    locals()["regularnet"+architecture].update({"batch_size":256})
    
#ablation
#replicate with set seed
architecture = "fast"
locals()["adacapnet"+architecture+"weightseed0npermut16"] = deepcopy(eval("adacapnet"+architecture))
locals()["adacapnet"+architecture+"weightseed0npermut16"]["hidden_params"].update({"initializer_params":{"init_seed":0}})
locals()["adacapnet"+architecture+"weightseed0npermut16"]["validation_fraction"] = False
locals()["adacapnet"+architecture+"weightseed0npermut16"]["early_stopping_criterion"] = False
locals()["adacapnet"+architecture+"weightseed0npermut1"] = deepcopy(eval("adacapnet"+architecture+"weightseed0npermut16"))
locals()["adacapnet"+architecture+"weightseed0npermut1"]["n_permut"] = 1
locals()["adacapnet"+architecture+"weightseed0npermut0"] = deepcopy(eval("adacapnet"+architecture+"weightseed0npermut16"))
locals()["adacapnet"+architecture+"weightseed0npermut0"]["n_permut"] = 0
locals()["regularnet"+architecture+"weightseed0"] = deepcopy(eval("adacapnet"+architecture+"weightseed0npermut0"))
locals()["regularnet"+architecture+"weightseed0"]["closeform_parameter_init"] = False
locals()["regularnet"+architecture+"weightseed0DO"] = deepcopy(eval("regularnet"+architecture+"weightseed0"))
locals()["regularnet"+architecture+"weightseed0DO"]["hidden_params"].update({"dropout":0.2})

#dependance dropout
dropout_range = [0., 0.1, 0.2, 0.3, 0.4, 0.5]
for architecture in ["fast","standard","resblock","batchstandard","batchresblock", "glu", "reglu", "resglu", "resreglu", "selu", "fastselu"]:
    for param_value in dropout_range:
        suffix = architecture+"dropout"+str(param_value).replace(".","_").replace("-","m")
        suffixbn = architecture+"dropoutbatchnorm"+str(param_value).replace(".","_").replace("-","m")
        locals()["adacapnet"+suffix] = deepcopy(eval("adacapnet"+architecture))
        locals()["adacapnet"+suffix]["hidden_params"].update({"dropout":param_value})
        locals()["adacapnet"+suffixbn] = deepcopy(eval("adacapnet"+suffix))
        locals()["adacapnet"+suffixbn]["hidden_params"].update({"batch_norm":True})
        
#dependance batch_size
batch_range = [16,32,64,128, 256,512,1024,2048]
for architecture in ["fast","standard","resblock", "wide", "resblockwide", "selu", "fastselu"]:
    for param_value in batch_range:
        locals()["adacapnet"+architecture+"bs"+str(param_value)] = deepcopy(eval("adacapnet"+architecture))
        locals()["adacapnet"+architecture+"bs"+str(param_value)].update({"batch_size":param_value, "epochs":True})
        locals()["regularnet"+architecture+"bs"+str(param_value)] = deepcopy(eval("adacapnet"+architecture+"bs"+str(param_value)))
        locals()["regularnet"+architecture+"bs"+str(param_value)]["hidden_params"].update({"dropout":0.2,"batch_norm":True})

#No time to compute    
#dependance width
width_range = [16,32,64,258,256,512,1024,2048]
for architecture in ["fast","standard","resblock","batchstandard","batchresblock", "glu", "reglu", "resglu", "resreglu", "selu", "fastselu"]:
    for param_value in width_range:
        locals()["adacapnet"+architecture+"width"+str(param_value)] = deepcopy(eval("adacapnet"+architecture))
        locals()["adacapnet"+architecture+"width"+str(param_value)]["hidden_params"].update({"width":param_value})
        
#dependance depth
depth_range = [1,2,3,4,5,6]
for architecture in ["fast","standard","resblock","batchstandard","batchresblock", "glu", "reglu", "resglu", "resreglu", "selu", "fastselu"]:
    for param_value in depth_range:
        locals()["adacapnet"+architecture+"depth"+str(param_value)] = deepcopy(eval("adacapnet"+architecture))
        locals()["adacapnet"+architecture+"depth"+str(param_value)]["hidden_params"].update({"depth":param_value})
        locals()["regularnet"+architecture+"depth"+str(param_value)] = deepcopy(eval("adacapnet"+architecture+"depth"+str(param_value)))
        locals()["regularnet"+architecture+"depth"+str(param_value)]["hidden_params"].update({"dropout":0.2,"batch_norm":True})
    
#dependance weight decay
weightdecay_range = [0.,1e-6,1e-5,1e-4,1e-3]
for architecture in ["fast","standard","resblock","batchstandard","batchresblock", "glu", "reglu", "resglu", "resreglu", "selu", "fastselu"]:
    for param_value in weightdecay_range:
        suffix = architecture+"weightdecay"+str(param_value).replace(".","_").replace("-","m")
        locals()["adacapnet"+suffix] = deepcopy(eval("adacapnet"+architecture))
        locals()["adacapnet"+suffix]["hidden_params"].update({"optimizer_params":{"weight_decay":param_value}})
        locals()["regularnet"+suffix] = deepcopy(eval("adacapnet"+suffix))
        locals()["regularnet"+suffix]["hidden_params"].update({"dropout":0.2,"batch_norm":True})