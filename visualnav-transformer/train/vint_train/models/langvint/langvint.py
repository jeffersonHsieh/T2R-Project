import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet
from vint_train.models.base_model import BaseModel
from vint_train.models.langvint.self_attention import MultiLayerDecoder
from transformers import CLIPModel

class LangViNT(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        late_fusion: Optional[bool] = False,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        LangViNT class: uses a Transformer-based architecture to encode (current and past) visual observations 
        and goals using an EfficientNet CNN, and predicts temporal distance and normalized actions 
        in an embodiment-agnostic manner
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
            obs_encoder (str): name of the EfficientNet architecture to use for encoding observations (ex. "efficientnet-b0")
            obs_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
        """
        super(LangViNT, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # take the output feature dimension of Linear layer in CLIP text projection head
        self.num_goal_features = self.clip_model.text_projection.out_features

        self.late_fusion = late_fusion
        # TODO: MULTIMODAL FUSION ENCODER!
        # TODO: Should we use CLIP-ViT as Observation encoder as well?
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            self.num_obs_features = self.obs_encoder._fc.in_features
            # if self.late_fusion:
            #     self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)
            # else:
            #     self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6) # obs+goal
            # # self.num_goal_features = self.goal_encoder._fc.in_features
        else:
            raise NotImplementedError
        
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
        # if self.num_goal_features != self.goal_encoding_size:
        #     self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        # else:
        # self.compress_goal_enc = nn.Identity()

        # train this only, always have this even if same feature size
        self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)

        self.decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size+2,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )


    def train(self, mode=True):
        super(LangViNT, self).train(mode=mode)
        # Overwrite train() to ensure Frozen models remain frozen.
        self.clip_model.eval()


    def forward(self, obs_img: torch.Tensor, goal_text_input: torch.Tensor,goal_text_attn_mask:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process observation images with EfficientNet
        # Split the observation into context based on the context size
        obs_img = torch.split(obs_img, 3, dim=1)
        obs_img = torch.concat(obs_img, dim=0)

        # Get the observation encoding
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)

        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)

        # Encode the goal text using CLIP
        
        goal_encoding = self.clip_model.get_text_features(
            input_ids=goal_text_input,
            attention_mask=goal_text_attn_mask
            )
        goal_encoding = self.compress_goal_enc(goal_encoding)
        # import pdb;pdb.set_trace()

        # Ensure goal_encoding is in the correct shape
        if len(goal_encoding.shape) == 2:
            goal_encoding = goal_encoding.unsqueeze(1)

        # Concatenate the goal encoding to the observation encoding
        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)

        # Pass through the decoder
        final_repr = self.decoder(tokens)

        # Predict distance and actions
        dist_pred = self.dist_predictor(final_repr)
        action_pred = self.action_predictor(final_repr)

        # Post-process action predictions
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # Convert position deltas into waypoints
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # Normalize the angle prediction

        # TODO: also return CLIP text projection pooled vector for regularization/alignment. should freeze CLIPTextEncoder and only unfreeze projection layer
        # TODO: finer grained alignment, maybe use GroundingDINO like object detectors for position+object alignment (both info available in generated caption)
        return dist_pred, action_pred