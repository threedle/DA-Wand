def create_model(opt):
    from .mesh_intseg import IntSegModel
    
    if opt.arch == "intseg":
        model = IntSegModel(opt)
    else: 
        raise ValueError(f"Unsupported architecture option: {opt.arch}")
    return model
