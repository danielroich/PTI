
def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
