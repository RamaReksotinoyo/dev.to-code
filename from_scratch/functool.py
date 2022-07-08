V=[1,2,3,4]
W=[1,2,3,4]

def vector_add(V,W):
    """adding two vector"""
    return [V+W for V,W in zip(V,W)]
