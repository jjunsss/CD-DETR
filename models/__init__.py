def get_models(model_name, args, num_classes):
    # build model
    if model_name == 'deform_detr':
        from .deform_detr import build_model
    elif model_name == 'dn_detr':
        from .dn_detr import build_model
    # elif model_name == ...:
    #     모델 계속 추가

    return build_model(args, num_classes)

def _prepare_denoising_args(model, targets, args=None, eval=False):
    if eval:
        dn_args = 0
    else:
        dn_args=(targets, args.scalar, args.label_noise_scale, args.box_noise_scale, args.num_patterns)
        if args.contrastive is not False:
            dn_args += (args.contrastive,)

    model.dn_args = dn_args # dn_detr & teacher_model도 고려?
    return model