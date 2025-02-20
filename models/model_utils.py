def unpack_feature_pyramid(feature_pyramid):
    try:
        # This works for resnets, efficient-nets
        [_, quarter, eights, _, out] = feature_pyramid
    except ValueError:
        # This works for convnexts
        [quarter, eights, _, out] = feature_pyramid
    return quarter, out