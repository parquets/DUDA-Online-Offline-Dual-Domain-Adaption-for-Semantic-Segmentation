from .discriminator import FCDiscriminator

def build_discriminator(config, in_channels):
    if config.network.discriminator is not None:
        return eval(config.network.discriminator)(in_channels)
    else:
        return None