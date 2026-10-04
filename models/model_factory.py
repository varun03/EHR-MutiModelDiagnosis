from models.lightweight_ehrknowgen import LightweightEHRKnowGen


def create_model(
    model_type,
    graph=None,
    class_prior=None,
    vocab_size=1,
    num_labels=1,
    hidden_dim=128,
    num_heads=4,
    num_layers=2,
    dropout=0.1,
    text_model_name="t5-small",
    calib_weight=0.5,
    pos_weight=1.0,
):
    """
    Factory method for creating models.

    Parameters
    ----------
    model_type : str
        "lightweight", "t5" or "knowgen"

    vocab_size : int
        Size of the input token vocabulary.

    num_labels : int
        Number of output classes.

    Returns
    -------
    torch.nn.Module
    """

    if model_type == "lightweight":
        return LightweightEHRKnowGen(
            vocab_size=vocab_size,
            num_labels=num_labels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

    elif model_type == "t5":
        from models.t5_ehrknowgen import T5EHRKnowGen

        return T5EHRKnowGen(
            model_name=text_model_name
        )

    elif model_type == "knowgen":
        from models.knowgen_t5 import KnowGenT5

        return KnowGenT5(
            graph=graph,
            class_prior=class_prior,
            model_name=text_model_name,
            num_heads=num_heads,
            dropout=dropout,
            calib_weight=calib_weight,
            pos_weight=pos_weight,
        )

    raise ValueError(
        f"Unknown model type: {model_type}"
    )