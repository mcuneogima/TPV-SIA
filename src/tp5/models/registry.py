"""
tp5.models.registry
===================
Registro central y constructores de modelos y componentes.
Permite instanciar componentes y modelos dinámicamente desde especificaciones YAML.
"""

from typing import Dict, Any, Optional
import numpy as np

from ..nn.activations import ACTIVATIONS, get_activation
from ..nn.losses import LOSSES, get_loss
from ..nn.optimizers import OPTIMIZERS, build_optimizer
from ..nn.schedules import SCHEDULES, build_schedule
from ..nn.init import INITIALIZERS, get_initializer
from ..nn.layers import Dense, Sequential

from .autoencoder import Autoencoder
from .vae import VAE


MODELS = {
    "autoencoder": Autoencoder,
    "basic": Autoencoder,
    "denoising": Autoencoder,
    "vae": VAE,
}


def build_model(
    cfg: Dict[str, Any],
    input_dim: int,
    rng: Optional[np.random.Generator] = None,
) -> Any:
    """
    Construye e inicializa un modelo (Autoencoder o VAE) a partir de la configuración.
    
    cfg: Diccionario completo de corrida conteniendo la sección 'model'.
    input_dim: Dimensión de entrada (ej. 35 para font, 768 para sprites).
    rng: Generador de números aleatorios para inicializaciones reproducibles.
    """
    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", "autoencoder").lower().strip()
    latent_dim = int(model_cfg.get("latent_dim", 2))
    init_name = model_cfg.get("init", "xavier")

    if model_type not in MODELS:
        raise ValueError(
            f"Tipo de modelo desconocido: '{model_type}'. "
            f"Opciones: {list(MODELS.keys())}"
        )

    # Construcción de decoder
    decoder_layers_cfg = model_cfg.get("decoder", {}).get("layers", [])
    decoder = Sequential()
    curr_dim = latent_dim
    for idx, lspec in enumerate(decoder_layers_cfg):
        out_dim = int(lspec["units"])
        act_name = lspec.get("activation", "identity")
        dense = Dense(
            curr_dim,
            out_dim,
            init=init_name,
            rng=rng,
            name=f"dec_dense_{idx}",
        )
        decoder.add(dense)
        if act_name != "identity":
            decoder.add(get_activation(act_name))
        curr_dim = out_dim
    output_dim = curr_dim

    # Si es VAE
    if model_type == "vae":
        beta = float(cfg.get("loss", {}).get("beta", 1.0))
        trunk_layers_cfg = model_cfg.get("encoder", {}).get("layers", [])
        encoder_trunk = Sequential()
        curr_dim = input_dim

        for idx, lspec in enumerate(trunk_layers_cfg):
            out_dim = int(lspec["units"])
            act_name = lspec.get("activation", "relu")
            dense = Dense(
                curr_dim,
                out_dim,
                init=init_name,
                rng=rng,
                name=f"enc_trunk_{idx}",
            )
            encoder_trunk.add(dense)
            if act_name != "identity":
                encoder_trunk.add(get_activation(act_name))
            curr_dim = out_dim

        mu_head = Dense(
            curr_dim,
            latent_dim,
            init=init_name,
            rng=rng,
            name="mu_head",
        )
        logvar_head = Dense(
            curr_dim,
            latent_dim,
            init=init_name,
            rng=rng,
            name="logvar_head",
        )

        return VAE(
            encoder_trunk=encoder_trunk,
            mu_head=mu_head,
            logvar_head=logvar_head,
            decoder=decoder,
            latent_dim=latent_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            beta=beta,
            name=cfg.get("run_name", "vae"),
        )

    # Si es Autoencoder (basic o denoising)
    encoder_layers_cfg = model_cfg.get("encoder", {}).get("layers", [])
    encoder = Sequential()
    curr_dim = input_dim
    for idx, lspec in enumerate(encoder_layers_cfg):
        out_dim = int(lspec["units"])
        act_name = lspec.get("activation", "identity")
        dense = Dense(
            curr_dim,
            out_dim,
            init=init_name,
            rng=rng,
            name=f"enc_dense_{idx}",
        )
        encoder.add(dense)
        if act_name != "identity":
            encoder.add(get_activation(act_name))
        curr_dim = out_dim

    return Autoencoder(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        input_dim=input_dim,
        output_dim=output_dim,
        name=cfg.get("run_name", "autoencoder"),
    )
