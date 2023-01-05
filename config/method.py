import imp
import rllib


from core.model_vectornet import ReplayBufferMultiAgentMultiWorker as ReplayBufferMultiAgent
from core.model_vectornet import ReplayBufferSingleAgentMultiWorker as ReplayBufferSingleAgent
from core.model_vectornet import RolloutBufferSingleAgentMultiWorker as RolloutBufferSingleAgent
# from core.model_vectornet import RolloutBufferSingleAgentWithCharacters as RolloutBufferSingleAgent
from core.model_vectornet import PointNetWithAgentHistoryOur
from core.model_vectornet import PointNetWithAgentHistory  ### no_character
from core.model_vectornet import PointNetWithCharacterAgentHistory  ### robust_character
from core.model_vectornet import PointNetWithCharactersAgentHistory  ### adaptive_character
from core.recognition_net import RecognitionNet, RecognitionNetNew,PointNetWithCharactersAgentHistoryRecog, RecognitionWoAttention, RecognitionNetSample

config_meta = rllib.basic.YamlConfig(
    device='cuda',
    num_cpus=1.0,
    num_gpus=0.2,
)


########################################################################
#### IndependentSAC ####################################################
########################################################################


config_isac__no_character = rllib.basic.YamlConfig(
    net_actor_fe=PointNetWithAgentHistory,
    net_critic_fe=PointNetWithAgentHistory,
    buffer=ReplayBufferMultiAgent,
    **config_meta.to_dict(),
)


config_isac__robust_character = rllib.basic.YamlConfig(
    net_actor_fe=PointNetWithCharacterAgentHistory,
    net_critic_fe=PointNetWithCharacterAgentHistory,
    buffer=ReplayBufferMultiAgent,
    **config_meta.to_dict(),
)


config_isac__adaptive_character = rllib.basic.YamlConfig(
    net_actor_fe=PointNetWithCharactersAgentHistory,
    net_critic_fe=PointNetWithCharactersAgentHistory,
    buffer=ReplayBufferMultiAgent,
    **config_meta.to_dict(),
)




########################################################################
#### SAC ###############################################################
########################################################################


config_sac = rllib.basic.YamlConfig(
    net_actor_fe=PointNetWithAgentHistory,
    net_critic_fe=PointNetWithAgentHistory,
    buffer=ReplayBufferSingleAgent,
    **config_meta.to_dict(),
)

config_sac_attn = rllib.basic.YamlConfig(
    net_actor_fe=PointNetWithAgentHistoryOur,
    net_critic_fe=PointNetWithAgentHistoryOur,
    buffer=ReplayBufferSingleAgent,
    **config_meta.to_dict(),
)


config_sac__adaptive_character = rllib.basic.YamlConfig(
    net_actor_fe=PointNetWithCharactersAgentHistory,
    net_critic_fe=PointNetWithCharactersAgentHistory,
    buffer=ReplayBufferSingleAgent,
    **config_meta.to_dict(),
)

########################################################################
#### PPO ###############################################################
########################################################################

config_ppo = rllib.basic.YamlConfig(
    net_ac_fe=PointNetWithAgentHistory,
    net_critic_fe=PointNetWithAgentHistory,
    buffer=RolloutBufferSingleAgent,
    net_ac = rllib.ppo.ActorCriticContinuous,
    **config_meta.to_dict(),
)

config_ppo_attn = rllib.basic.YamlConfig(
    net_ac_fe=PointNetWithAgentHistoryOur,
    net_critic_fe=PointNetWithAgentHistoryOur,
    buffer=RolloutBufferSingleAgent,
    net_ac = rllib.ppo.ActorCriticContinuous,
    **config_meta.to_dict(),
)

########################################################################
#### IndependentSAC_recog ##############################################
########################################################################

config_isac_recog = rllib.basic.YamlConfig(
    net_actor_fe=RecognitionNetNew,
    net_critic_fe=PointNetWithCharactersAgentHistory,
    # net_actor_recog=RecognitionNet,
    # net_critic_recog=RecognitionNet,
    buffer=ReplayBufferSingleAgent,
    **config_meta.to_dict(),
)
config_isac_recog_woattn = rllib.basic.YamlConfig(
    net_actor_fe=PointNetWithCharactersAgentHistoryRecog,
    net_critic_fe=PointNetWithCharactersAgentHistoryRecog,
    net_actor_recog=RecognitionWoAttention,
    net_critic_recog=RecognitionWoAttention,
    buffer=ReplayBufferSingleAgent,
    **config_meta.to_dict(),
)
config_supervise = rllib.basic.YamlConfig(
    net_actor_fe=PointNetWithCharactersAgentHistoryRecog,
    net_recog=RecognitionNet,
    buffer=ReplayBufferSingleAgent,
    **config_meta.to_dict(),
)

config_supervise_sample = rllib.basic.YamlConfig(
    net_actor_fe=PointNetWithCharactersAgentHistoryRecog,
    net_recog=RecognitionNetSample,
    buffer=ReplayBufferSingleAgent,
    **config_meta.to_dict(),
)

config_supervise_roll = rllib.basic.YamlConfig(
    net_actor_fe=PointNetWithCharactersAgentHistoryRecog,
    net_recog=RecognitionNet,
    buffer=RolloutBufferSingleAgent,
    **config_meta.to_dict(),
)

config_woattn = rllib.basic.YamlConfig(
    net_actor_fe=PointNetWithCharactersAgentHistoryRecog,
    net_recog=RecognitionWoAttention,
    buffer=ReplayBufferSingleAgent,
    **config_meta.to_dict(),
)

