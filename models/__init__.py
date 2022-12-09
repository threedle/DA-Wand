def create_model(opt):
    from .dawand import DAWand
    
    if opt.arch == "meshcnn":
        model = DAWand(opt)
    else: 
        raise ValueError(f"Unsupported architecture option: {opt.arch}")
    return model
