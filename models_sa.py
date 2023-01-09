
from statistics import mode
from charset_normalizer import models

from ray import method
from models_ma import ModelPath
import rllib


class sac__bottleneck__adaptive_adv_background(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvSingleAgentAdv/2022-09-08-21:58:42----ray_sac__bottleneck__adaptive_adv_background--parallel/saved_models_method'
    model_dir_adv = '~/github/zdk/recognition-rl/results/SAC-EnvSingleAgentAdv/2022-09-08-21:58:42----ray_sac__bottleneck__adaptive_adv_background--parallel/saved_models_method_adv'
    model_num = 173800  ### bad





################################################################################
###### bottleneck ##############################################################
################################################################################


class sac__bottleneck__idm(ModelPath):
    method = 'sac'

    # model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvInteractiveSingleAgent/2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario/saved_models_method'
    # model_num = 735800

    
    # model_dir = "~/github/zdk/recognition-rl/results/SAC-EnvInteractiveSingleAgent/2022-10-07-09:39:04----Nothing/saved_models_method"
    # model_num = 14000

    
    model_dir = "~/github/zdk/recognition-rl/results/SAC-EnvInteractiveSingleAgent/2022-10-08-18:41:38----Nothing/saved_models_method"
    model_num = 268800
class sac__bottleneck__no_character(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvSingleAgentAdv/2022-09-08-21:58:42----ray_sac__bottleneck__adaptive_adv_background--parallel/saved_models_method'
    model_num = 173800


class sac__bottleneck__robust_copo(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvSingleAgentAdv/2022-09-08-21:58:42----ray_sac__bottleneck__adaptive_adv_background--parallel/saved_models_method'
    model_num = 173800

class sac__bottleneck__explicit_adv(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvSingleAgentAdv/2022-09-08-21:58:42----ray_sac__bottleneck__adaptive_adv_background--parallel/saved_models_method'
    model_num = 173800



class sac__bottleneck__adaptive(ModelPath):
    method = 'sac'

    # model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvInteractiveSingleAgent/2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario/saved_models_method'
    # model_num = 888400

    # model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvInteractiveSingleAgent/2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario/saved_models_method'
    # model_num = 888400
    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvInteractiveSingleAgent/2022-10-14-23:29:07----Nothing/saved_models_method'
    model_num = 98800

class sac__bottleneck__duplicity(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvSingleAgentAdv/2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background/saved_models_method'
    model_num = 318800

    model_num = 319400  ### high iid index


class ppo__bottleneck__adaptive(ModelPath):
    method = 'ppo'

    model_dir = '~/github/zdk/recognition-rl/results/PPO-EnvInteractiveSingleAgent/2022-10-06-13:13:26----Nothing/saved_models_method'
    model_num = 7

    

class ppo__bottleneck__idm(ModelPath):
    method = 'ppo'

    model_dir = '~/github/zdk/recognition-rl/results/PPO-EnvInteractiveSingleAgent/2022-10-07-14:57:55----Nothing/saved_models_method'
    model_num = 13559



class isac__bottleneck__robust_copo(ModelPath):
    method = 'IndependentSAC_v0'
    
    #xue in 2022.10.10 test in 116 server
    model_dir = '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario/saved_models_method'
    model_num = 568800

class isac__bottleneck__adaptive(ModelPath):
    method = 'IndependentSAC_v0'

    #2022.10.10 test in 116 server
    model_dir = '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method'
    model_num = 865800
    #2022/12/21
    model_dir = '~/github/zdk/recognition-rl/models/origin_no_history_bottleneck'
    model_num = 445600

class isac_recog__bottleneck__idm(ModelPath):
    # method = 'IndependentSAC_v0'

    method = 'IndependentSAC_recog'
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_recog-EnvInteractiveSingleAgent/2022-10-13-19:00:02----Nothing/saved_models_method'
    model_num = 48800

class isac_recog__bottleneck__adaptive(ModelPath):
    # method = 'IndependentSAC_v0'

    method = 'IndependentSAC_recog'


    # model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_recog-EnvInteractiveSingleAgent/2022-10-24-20:58:05----Nothing--isac_recog__adaptive/saved_models_method'
    # model_num = 130000
    # model_num = 134000
    # model_num = 100000
    # model_num = 116000
    # model_num = 128800
    # model_num = 42600
    # model_num = 139800
    # model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_recog-EnvInteractiveSingleAgent/2022-10-26-14:40:58----Nothing--isac_recog__adaptive/saved_models_method'
    # model_num = 120000
    # model_num = 130000
    # model_num = 140000
    # model_num = 110000
    # model_num = 100000
    # model_num = 90000
    # model_num = 80000
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_recog-EnvInteractiveSingleAgent/2022-10-27-19:11:02----Nothing--isac_recog__adaptive/saved_models_method'
    # model_num = 139400
    # model_num = 140000
    # model_num = 130000
    # model_num = 120000
    # model_num = 110000
    # model_num = 100000
    # model_num = 125000
    # model_num = 90000
    model_num = 10000
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_recog-EnvInteractiveSingleAgent/2022-10-28-21:18:26----Nothing--isac_recog__adaptive/saved_models_method'
    model_num = 149800
    model_num = 140000
    model_num = 130000
    model_num = 120000
    model_num = 110000
    model_num = 100000
    model_num = 90000
    model_num = 80000
    model_num = 200
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_recog-EnvInteractiveSingleAgent/2022-10-29-16:18:08----Nothing--isac_recog__adaptive/saved_models_method'
    model_num = 25800

    # #2022/12/23
    # model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_recog-EnvInteractiveSingleAgent/2022-12-23-13:31:17----Nothing--isac_recog__downsample_adaptive/saved_models_method'
    # model_num = 280000
    #2022/12/30
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_recog-EnvInteractiveSingleAgent/2022-12-28-21:59:09----Nothing--isac_recog__new_adaptive/saved_models_method'
    model_num = 211000
    # #2023/1/1
    # model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_recog-EnvInteractiveSingleAgent/2022-10-27-19:11:02----Nothing--isac_recog__adaptive/saved_models_method'
    # model_num = 99800
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_recog-EnvInteractiveSingleAgent/2023-01-02-19:47:47----Nothing--isac_recog__new_adaptive/saved_models_method'
     
    model_num = 19000
    # model_num= 53000
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_recog-EnvInteractiveSingleAgent/2023-01-03-19:21:50----Nothing--isac_recog__new_adaptive/saved_models_method/'
    #89.0
    model_num = 192000
    model_num = 220000
    model_num = 230000

class isac_recog_woattn__bottleneck__adaptive(ModelPath):
    method = 'ISAC_recog_woattn'

    model_dir = '~/github/zdk/recognition-rl/results/ISAC_recog_woattn-EnvInteractiveSingleAgent/2023-01-03-19:56:43----Nothing--isac_recog_woattn__new_adaptive/saved_models_method'
    model_num = 202000

class isac_recog__bottleneck__robust(ModelPath):

    method = 'IndependentSAC_recog'
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_recog-EnvInteractiveSingleAgent/2022-10-16-13:37:05----Nothing--isac_recog__robust/saved_models_method'

    model_num = 122800

class isac_recog__bottleneck__no_character(ModelPath):

    method = 'IndependentSAC_recog'
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_recog-EnvInteractiveSingleAgent/2022-10-17-14:47:43----Nothing--isac_recog__no_character/saved_models_method'

    model_num = 133000

class supervise__bottleneck__adaptive(ModelPath):

    method = 'IndependentSAC_supervise'
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_supervise-EnvInteractiveSingleAgent/2022-11-20-00:46:32----Nothing--supervise-hrz30-act10/saved_models_method'
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_supervise-EnvInteractiveSingleAgent/2022-11-20-17:19:13----Nothing--supervise-hrz30-act10/saved_models_method'
    #2022-11-19
    method = 'IndependentSACsupervise'
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveSingleAgent/2022-11-26-12:32:44----Nothing--supervise-hrz30-act10/saved_models_method'    
    model_num = 312000
    model_num = 224000
    model_num = 187000
    model_num = 300000
    
    model_num = 700000

    model_num = 2900000
    model_num = 2800000
    model_num = 2700000


class supervise__bottleneck__adaptive__given_number(ModelPath):
    def update(self, config: rllib.basic.YamlConfig, model_num):
        self.model_num = model_num
        config.set('method', self.method)
        config.set('model_dir', self.model_dir)
        config.set('model_num', self.model_num)
        if config.model_index != None:
            config.set('model_num',  config.model_index)
        model_exp = config.model_dir.split('/')[-2]
        config.set('description', config.description + f'--from--{model_exp}-{config.model_num}')
    method = 'IndependentSACsupervise'
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveSingleAgent/2022-11-26-12:32:44----Nothing--supervise-hrz30-act10/saved_models_method'
    
    # model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveSingleAgent/2022-11-30-15:05:17----Nothing--supervise-hrz10-act10/saved_models_method'

class supervise_sampling__bottleneck__adaptive__given_number(ModelPath):
    def update(self, config: rllib.basic.YamlConfig, model_num):
        self.model_num = model_num
        config.set('method', self.method)
        config.set('model_dir', self.model_dir)
        config.set('model_num', self.model_num)
        if config.model_index != None:
            config.set('model_num',  config.model_index)
        model_exp = config.model_dir.split('/')[-2]
        config.set('description', config.description + f'--from--{model_exp}-{config.model_num}')
    method = 'IndependentSACsupervise'
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveSingleAgent/2022-12-05-13:05:38----Nothing--supervise-sampling-hrz10-act10/saved_models_method'
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_recog-EnvInteractiveSingleAgent/2022-12-26-22:01:02----Nothing--isac_recog__downsample_new_adaptive/saved_models_method'

class recog_rl__bottleneck__adaptive__given_number(ModelPath):
    def update(self, config: rllib.basic.YamlConfig, model_num):
        self.model_num = model_num
        config.set('method', self.method)
        config.set('model_dir', self.model_dir)
        config.set('model_num', self.model_num)
        if config.model_index != None:
            config.set('model_num',  config.model_index)
        model_exp = config.model_dir.split('/')[-2]
        config.set('description', config.description + f'--from--{model_exp}-{config.model_num}')
    method = 'IndependentSAC_recog'
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_recog-EnvInteractiveSingleAgent/2022-12-26-22:01:02----Nothing--isac_recog__downsample_new_adaptive/saved_models_method'
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_recog-EnvInteractiveSingleAgent/2022-12-28-21:59:09----Nothing--isac_recog__new_adaptive/saved_models_method'


################################################################################
###### intersection ############################################################
################################################################################


class sac__intersection__idm(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvInteractiveSingleAgent/2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario/saved_models_method'
    model_num = 735800

    model_num = 755400



class sac__intersection__no_character(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvInteractiveSingleAgent/2022-09-16-21:18:51----ray_sac__no_character__multi_scenario/saved_models_method'
    model_num = 393200

    model_num = 735800   ### -


class sac__intersection__robust_copo(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvInteractiveSingleAgent/2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario/saved_models_method'
    model_num = 412200

    model_num = 584800


class sac__intersection__explicit_adv(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvSingleAgentAdv/2022-09-08-21:58:42----ray_sac__intersection__adaptive_adv_background--parallel/saved_models_method'



class sac__intersection__adaptive(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvInteractiveSingleAgent/2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario/saved_models_method'
    model_num = 1114000


class sac__intersection__duplicity(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvSingleAgentAdv/2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background/saved_models_method'
    model_num = 318800

    model_num = 319400  ### high iid index

    model_num = 319200








################################################################################
###### merge ###################################################################
################################################################################


class sac__merge__idm(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvInteractiveSingleAgent/2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario/saved_models_method'
    model_num = 700600



class sac__merge__no_character(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvInteractiveSingleAgent/2022-09-16-21:18:51----ray_sac__no_character__multi_scenario/saved_models_method'
    model_num = 610000


class sac__merge__robust_copo(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvInteractiveSingleAgent/2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario/saved_models_method'
    model_num = 631200


class sac__merge__explicit_adv(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvSingleAgentAdv/2022-09-08-21:58:42----ray_sac__merge__adaptive_adv_background--parallel/saved_models_method'



class sac__merge__adaptive(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvInteractiveSingleAgent/2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario/saved_models_method'
    model_num = 1016000

    model_num = 696800


class sac__merge__duplicity(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvSingleAgentAdv/2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background/saved_models_method'
    model_num = 318800

    model_num = 319400
    model_num = 487000

    model_num = 580000







################################################################################
###### roundabout ############################################################
################################################################################


class sac__roundabout__idm(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvInteractiveSingleAgent/2022-09-14-17:17:56----ray_sac__idm_background__multi_scenario/saved_models_method'
    model_num = 735800



class sac__roundabout__no_character(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvInteractiveSingleAgent/2022-09-16-21:18:51----ray_sac__no_character__multi_scenario/saved_models_method'
    model_num = 393200


class sac__roundabout__robust_copo(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvInteractiveSingleAgent/2022-09-16-21:31:35----ray_sac__robust_character_copo__multi_scenario/saved_models_method'
    model_num = 412200


class sac__roundabout__explicit_adv(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvSingleAgentAdv/2022-09-08-21:58:42----ray_sac__roundabout__adaptive_adv_background--parallel/saved_models_method'



class sac__roundabout__adaptive(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvInteractiveSingleAgent/2022-09-15-19:14:57----ray_sac__adaptive_character__multi_scenario/saved_models_method'


class sac__roundabout__duplicity(ModelPath):
    method = 'sac'

    model_dir = '~/github/zdk/recognition-rl/results/SAC-EnvSingleAgentAdv/2022-09-15-19:16:14----ray_sac__multi_scenario__adaptive_adv_background/saved_models_method'
    model_num = 318800

    model_num = 319400  ### high iid index


class svo_as_action__bottleneck__adaptive(ModelPath):

    method = 'RecogV1'
    model_dir = '~/github/zdk/recognition-rl/results/RecogV1-EnvInteractiveSingleAgent/2023-01-07-20:57:41----Nothing--isac_recog__new_action/saved_models_method'
    #so so
    model_num = 58000

    model_dir = '~/github/zdk/recognition-rl/results/RecogV1-EnvInteractiveSingleAgent/2023-01-09-00:01:10----Nothing--isac_recog__new_action/saved_models_method'
    model_num = 92000
    model_num = 100000

class svos_as_action__bottleneck__adaptive(ModelPath):

    method = 'RecogV2'
    model_dir = '~/github/zdk/recognition-rl/results/RecogV2-EnvInteractiveSingleAgent/2023-01-07-20:16:44----Nothing--recog__dynamic_action/saved_models_method'
    #so so
    model_num = 92000
    
    model_num = 94000
    
    model_num = 96000

    model_num = 98000