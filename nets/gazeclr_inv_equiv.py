import torch
import torch.nn as nn
from losses.contrastive_loss import ContrastiveLoss
import torchvision.models as models


class BaseSimCLRException(Exception):
    """Base exception"""


class InvalidBackboneError(BaseSimCLRException):
    """Raised when the choice of backbone Convnet is invalid."""


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GazeCLRInvEq(nn.Module):

    def __init__(self, config):
        super(GazeCLRInvEq, self).__init__()

        base_model = config.arch
        out_dim = config.out_dim
        self.substring = 'face'

        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = Identity()

        self.dim_mlp = dim_mlp

        # Invariance projector p_1():  z_i = g(h_i) = W(2)σ(W(1)h_i)
        self.projector_inv = nn.Sequential(
            nn.Linear(self.dim_mlp, config.projection_dim, bias=False),
            nn.ReLU(),
            nn.Linear(config.projection_dim, out_dim, bias=False),
        )

        # Equivariance projector p_2():  z_i = g(h_i) = W(2)σ(W(1)h_i)
        self.projector_equiv = nn.Sequential(
            nn.Linear(self.dim_mlp, config.projection_dim, bias=False),
            nn.ReLU(),
            nn.Linear(config.projection_dim, out_dim, bias=False),
        )

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def get_embeddings(self, x, last_layer=False):
        if last_layer is True:
            return self.projector_inv(self.backbone(x))
        else:
            return self.backbone(x)

    def forward_pass_equiv(self, imgs, rot1, rot2):
        batch_size = imgs.size(0)
        h_i = self.backbone(imgs)
        z_i = self.projector_equiv(h_i)
        z_i = z_i.view(batch_size, 3, -1)
        z_i = torch.bmm(rot1, z_i)
        z_i = torch.bmm(rot2, z_i)
        z_i = z_i.view(z_i.size()[0], -1)

        return h_i, z_i

    def forward_pass_inv(self, imgs):
        h_i = self.backbone(imgs)
        z_i = self.projector_inv(h_i)
        return z_i

    def forward(self, data_dict):

        output_dict = {}
        # equivarient representations
        positive_images = data_dict['img_a']
        invariant_images = data_dict['inv_a']
        R_inv_gaze = data_dict['{}_R'.format(self.substring)]
        R_relative_cam = data_dict['camera_transformation'][:, :, :3, :3]
        assert len(positive_images.size()) == 5

        b, n, c, h, w = positive_images.size()
        assert R_inv_gaze.shape == (b, n, 3, 3)
        assert R_relative_cam.shape == (b, n, 3, 3)

        R_inv_gaze = torch.transpose(R_inv_gaze.view(b*n, 3, 3), 1, 2)
        R_relative_cam = torch.transpose(R_relative_cam.view(b*n, 3, 3), 1, 2)

        h_,  z_ = self.forward_pass_equiv(positive_images.view(b*n, c, h, w), R_inv_gaze, R_relative_cam)

        output_dict['feat'] = h_
        output_dict['equiv_proj'] = z_.view(b, n, -1)

        output_dict['inv_feat_a'] = self.projector_inv(h_).view(b, n, -1)
        output_dict['inv_feat_b'] = self.forward_pass_inv(invariant_images.view(b*n, c, h, w)).view(b, n, -1)

        return output_dict


class TrainGazeCLRInvEq(object):

    def __init__(self, config, device):

        self.device = device
        self.n_views = config.num_positives + 1
        self.batch_size = config.batch_size

        self.args = config

        self.model = GazeCLRInvEq(config).to(device)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.criterion = ContrastiveLoss(device, temperature=config.temperature)

    def compute_losses(self, input_dict):

        loss_dict = {}

        output_dict = self.model(input_dict)

        equivariant_features = output_dict['equiv_proj']
        invariant_features_a = output_dict['inv_feat_a']
        invariant_features_b = output_dict['inv_feat_b']

        assert equivariant_features.size()[1] == self.n_views
        assert invariant_features_a.size()[1] == self.n_views
        assert invariant_features_b.size()[1] == self.n_views

        # equivariant loss
        equivariant_contrastive_loss = 0
        count = 0
        for v1 in range(self.n_views):
            for v2 in range(v1+1, self.n_views):

                _loss = self.criterion(equivariant_features[:, v1], equivariant_features[:, v2])
                equivariant_contrastive_loss += _loss
                count += 1
        equivariant_contrastive_loss /= count

        # invariant loss
        invariant_contrastive_loss = 0
        count = 0
        for v1 in range(self.n_views):
            _loss = self.criterion(invariant_features_a[:, v1], invariant_features_b[:, v1])
            invariant_contrastive_loss += _loss
            count += 1
        invariant_contrastive_loss /= count

        loss_dict['equivariant_contrastive_loss'] = equivariant_contrastive_loss
        loss_dict['invariant_contrastive_loss'] = invariant_contrastive_loss
        loss_dict['total_loss'] = invariant_contrastive_loss + equivariant_contrastive_loss

        return loss_dict, output_dict

