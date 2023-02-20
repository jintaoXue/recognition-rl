from pyexpat import model
from charset_normalizer import models
from ray import method
import rllib



class ModelPath(object):
    method = 'none'

    model_dir = '~'
    model_num = -1

    def __init__(self):
        pass

    def update(self, config: rllib.basic.YamlConfig):
        config.set('method', self.method)
        config.set('model_dir', self.model_dir)
        config.set('model_num', self.model_num)
        if config.model_index != None:
            config.set('model_num',  config.model_index)
        model_exp = config.model_dir.split('/')[-2]
        config.set('description', config.description + f'--from--{model_exp}-{config.model_num}')





############################################################################
#### no character ##########################################################
############################################################################




class isac_no_character__multi_scenario(ModelPath):
    method = 'independentsac_v0'

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-13-22:44:21----ray_isac_no_character__multi_scenario/saved_models_method'
    model_num = 696800

    ### final
    model_dir = '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-13-22:44:21----ray_isac_no_character__multi_scenario/saved_models_method'
    model_num = 696800




class isac_no_character__bottleneck(ModelPath):
    method = 'independentsac_v0'

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-13-22:44:21----ray_isac_no_character__multi_scenario/saved_models_method'
    model_num = 696800  ### very bad, interesting
    model_num = 697000

    ### final
    model_dir = '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-13-22:44:21----ray_isac_no_character__multi_scenario/saved_models_method'
    model_num = 696400






############################################################################
#### robust character ######################################################
############################################################################


class isac_robust_character__bottleneck(ModelPath):
    method = 'independentsac_v0'

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario/saved_models_method'
    model_num = 567600

    model_num = 568600  ### good but not our story


    ### final
    model_dir = '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario/saved_models_method'
    model_num = 568800


class isac_robust_character__intersection(ModelPath):
    method = 'independentsac_v0'

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario/saved_models_method'
    model_num = 593600


    ### final
    model_dir = '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario/saved_models_method'
    model_num = 593800



class isac_robust_character__merge(ModelPath):
    method = 'independentsac_v0'

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario/saved_models_method'
    model_num = 541000

    ### final
    model_dir = '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario/saved_models_method'
    model_num = 541000



class isac_robust_character__roundabout(ModelPath):
    method = 'independentsac_v0'

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario/saved_models_method'
    model_num = 580000

    model_num = 579200  ### good but not our story

    ### final
    model_dir = '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-13-22:44:39----ray_isac_robust_character__multi_scenario/saved_models_method'
    model_num = 579800



class isac_robust_character_copo__multi_scenario(ModelPath):
    method = 'independentsac_v0'

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-16-00:35:52----ray_isac_robust_character_copo__multi_scenario/saved_models_method'
    model_num = 126000









############################################################################
#### adaptive character ####################################################
############################################################################

class isac_adaptive_character__bottleneck(ModelPath):
    method = 'independentsac_v0'

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck/saved_models_method'
    model_num = 595800  ### bad
    model_num = 596000
    model_num = 596200
    model_num = 596400
    model_num = 596600
    model_num = 596800  ### bad
    model_num = 597000  ### bad
    model_num = 597200  ### good before story
    model_num = 597400  ### bad
    model_num = 597600  ### bad
    model_num = 597800  ### bad
    model_num = 598000  ### bad
    model_num = 598200  ### bad
    model_num = 598400  ### bad
    model_num = 598600  ### bad
    model_num = 598800  ### bad
    model_num = 599000  ### bad
    model_num = 599200  ### bad
    model_num = 599400  ### bad
    model_num = 599600  ### bad
    model_num = 599800  ### bad
    model_num = 600000  ### bad
    model_num = 600200  ### bad
    model_num = 600400  ### bad
    model_num = 600600  ### just so so
    model_num = 601000  ### good, maybe good diversity
    model_num = 601200  ### bad
    model_num = 601400  ### bad
    model_num = 601600  ### bad
    model_num = 601800  ### just so so
    model_num = 602000  ### bad
    model_num = 602200  ### bad story
    model_num = 602400  ### bad story
    model_num = 602600  ### bad
    model_num = 602800  ### bad
    model_num = 603000  ### bad
    model_num = 603200  ### bad
    model_num = 603400  ### bad

    model_num = 602800  ### bad
    model_num = 603000  ### bad
    model_num = 603200  ### bad
    model_num = 603400  ### bad

    model_num = 944400  ### bad
    model_num = 944600  ### bad
    model_num = 944800  ### bad
    model_num = 945000  ### bad
    model_num = 945200  ### very bad
    model_num = 945400  ### bad
    model_num = 945600  ### just so so
    model_num = 945800  ### good, low off road
    model_num = 946000  ### bad
    model_num = 946200  ### just so so
    model_num = 946400  ### bad

    model_num = 1285800  ### bad
    model_num = 1286000  ### bad
    model_num = 1286200  ### bad
    model_num = 1286400  ### bad

    # model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-03-21:01:23----ray_isac_adaptive_character__bottleneck/saved_models_method'
    # model_num = 1077000  ### not good story
    # model_num = 1077000  ### not good story
    # model_num = 1077400  ### bad
    # model_num = 1077600  ### bad
    # model_num = 1078000  ### bad
    # model_num = 1078200  ### bad
    # model_num = 1078400  ### bad
    # model_num = 1078600  ### not good story




    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-10-22:42:06----ray_isac_adaptive_character__multi_scenario/saved_models_method'

    model_num = 705800  ### bad
    model_num = 706000  ### bad
    model_num = 706200  ### bad
    model_num = 706400  ### bad
    
    model_num = 718200  ### bad
    model_num = 718400  ### bad
    model_num = 718600  ### just so so
    model_num = 718800  ### bad

    model_num = 719000  ### bad
    model_num = 719200  ### good
    model_num = 719400  ### bad
    model_num = 719600  ### bad

    model_num = 719800  ### bad
    model_num = 720000  ### bad
    model_num = 720200  ### bad
    model_num = 720400  ### bad



    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-10-22:40:25----ray_isac_adaptive_character__multi_scenario/saved_models_method'
    model_num = 580000
    model_num = 581000
    model_num = 582000
    model_num = 583000

    model_num = 592200  ### bad
    model_num = 592400  ### bad
    model_num = 592600  ### bad
    model_num = 592800  ### bad
    model_num = 593000  ### just so so
    model_num = 593200  ### bad
    model_num = 593400  ### bad
    model_num = 593600  ### bad

    model_num = 716800  ### bad
    model_num = 717000  ### bad
    model_num = 717200  ### bad
    model_num = 717400  ### bad
    model_num = 717600  ### bad
    model_num = 717800  ### bad
    model_num = 718000  ### bad
    model_num = 718200  ### bad

    model_num = 973000  ### bad
    model_num = 973200  ### bad
    model_num = 973400  ### bad
    model_num = 973600  ### bad
    model_num = 973800  ### bad
    model_num = 974000  ### bad
    model_num = 974200  ### good but not good story, for qualitative
    model_num = 974400  ### good but not good story, for qualitative
    model_num = 974600  ### bad
    model_num = 974800  ### good but not good story, for qualitative
    model_num = 975000  ### bad
    model_num = 975200  ### bad



    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method'
    
    model_num = 853000  ### bad
    model_num = 853200  ### bad
    model_num = 853400  ### bad
    model_num = 853600  ### bad
    model_num = 853800  ### good but not enough
    model_num = 854000  ### bad
    model_num = 854200  ### bad
    model_num = 854400  ### bad


    model_num = 859000  ### bad
    model_num = 859200  ### bad
    model_num = 859400  ### bad
    model_num = 859600  ### bad
    model_num = 859800  ### bad
    model_num = 860000  ### bad
    model_num = 860200  ### good
    model_num = 860400  ### bad
    model_num = 860600  ### bad
    model_num = 860800  ### bad
    model_num = 861000  ### bad
    model_num = 861200  ### bad
    model_num = 861400  ### bad
    model_num = 861600  ### bad
    model_num = 861800  ### bad
    model_num = 862000  ### bad
    model_num = 862200  ### bad
    model_num = 862400  ### bad
    model_num = 862600  ### bad
    model_num = 862800  ### bad

    model_num = 863000  ### bad
    model_num = 863200  ### bad
    model_num = 863400  ### good but not good story, for qualitative
    model_num = 863600  ### bad
    model_num = 863800  ### bad
    
    model_num = 864000  ### good !!!
    model_num = 864200  ### bad
    model_num = 864400  ### bad
    model_num = 864600  ### bad
    model_num = 864800  ### bad
    
    model_num = 865000  ### bad
    model_num = 865200  ### bad
    model_num = 865400  ### good
    model_num = 865600  ### bad
    model_num = 865800  ### good !!!
    
    model_num = 866000  ### just so so
    model_num = 866200  ### just so so
    model_num = 866400  ### bad
    model_num = 866600  ### bad
    model_num = 866800  ### bad
    
    model_num = 867000  ### bad
    model_num = 867200  ### bad
    model_num = 867400  ### bad
    model_num = 867600  ### bad
    model_num = 867800  ### just so so




    ### final
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method'
    model_num = 865800  ### good !!!





class isac_adaptive_character__intersection__old(ModelPath):
    method = 'independentsac_v0'

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-07-23:19:44----ray_isac_adaptive_character__intersection/saved_models_method'
    model_num = 359600  ### low failure
    model_num = 359800  ### bad
    model_num = 360000  ### good
    model_num = 360200  ### bad
    model_num = 360400  ### just so so
    model_num = 360600  ### maybe good
    model_num = 360800  ### bad
    model_num = 361000  ### good before story

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-07-23:20:15----ray_isac_adaptive_character__intersection/saved_models_method'
    model_num = 528000

    model_num = 527200  ### just so so
    model_num = 527400  ### not good story
    model_num = 527600  ### good, best up to now, but high off route
    model_num = 527800  ### good, low failure
    model_num = 528000  ### good
    model_num = 528200  ### bad
    model_num = 528400  ### just so so
    model_num = 528600  ### high off road

    model_num = 614000  ### just so so
    model_num = 614200  ### bad
    model_num = 614400  ### just so so
    model_num = 614600  ### good
    model_num = 614800  ### bad
    model_num = 615000  ### bad
    model_num = 615200  ### bad



    model_num = 527800  ### good, low failure
    model_num = 614600  ### good




    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-10-22:40:25----ray_isac_adaptive_character__multi_scenario/saved_models_method'
    
    model_num = 423000  ### bad
    model_num = 423200  ### bad
    model_num = 423400  ### bad
    model_num = 423600  ### bad
    model_num = 423800  ### good but not very good story
    model_num = 424000  ### bad
    model_num = 424200  ### bad
    model_num = 424400  ### bad

    model_num = 531400  ### bad
    model_num = 531600  ### bad
    model_num = 531800  ### bad
    model_num = 532000  ### bad


    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-10-22:42:06----ray_isac_adaptive_character__multi_scenario/saved_models_method'
    model_num = 202400  ### ! tmp

    model_num = 517200  ### bad
    model_num = 517400  ### just so so
    model_num = 517600  ### bad
    model_num = 517800  ### just so so
    model_num = 518000  ### bad
    model_num = 518200  ### bad
    model_num = 518400  ### bad
    model_num = 518600  ### bad
    model_num = 518800  ### bad
    model_num = 519000  ### bad
    model_num = 519200  ### bad
    model_num = 519400  ### bad

    model_num = 551600  ### bad
    model_num = 551800  ### bad
    model_num = 552000  ### bad
    model_num = 552200  ### bad


    model_num = 705000  ### bad
    model_num = 705200  ### bad
    model_num = 705400  ### bad
    model_num = 705600  ### bad
    model_num = 705800  ### bad
    model_num = 706000  ### bad
    model_num = 706200  ### bad
    model_num = 706400  ### bad
    model_num = 706600  ### good but not very good story
    model_num = 706800  ### bad
    model_num = 707000  ### bad
    model_num = 707200  ### bad
    model_num = 707400  ### bad
    model_num = 707600  ### bad
    model_num = 707800  ### good but not very good story
    model_num = 708000  ### bad
    model_num = 708200  ### bad
    model_num = 708400  ### bad
    model_num = 708600  ### bad
    model_num = 708800  ### bad
    model_num = 709000  ### bad
    model_num = 709200  ### bad
    model_num = 709400  ### bad
    model_num = 709600  ### bad
    model_num = 709800  ### bad
    model_num = 710000  ### bad
    model_num = 710200  ### bad
    model_num = 710400  ### bad
    model_num = 710600  ### bad
    model_num = 710800  ### bad
    model_num = 711000  ### bad
    model_num = 711200  ### bad
    model_num = 711400  ### bad
    model_num = 711600  ### bad
    model_num = 711800  ### bad
    model_num = 712000  ### bad
    model_num = 712200  ### bad
    model_num = 712400  ### bad
    model_num = 712600  ### bad
    model_num = 712800  ### bad
    
    model_num = 713000  ### bad
    model_num = 713200  ### bad
    model_num = 713400  ### bad
    model_num = 713600  ### good but not very good story
    model_num = 713800  ### good but not very good story
    model_num = 714000  ### bad
    model_num = 714200  ### bad
    model_num = 714400  ### bad
    model_num = 714600  ### bad
    model_num = 714800  ### bad
    model_num = 715000  ### bad
    model_num = 715200  ### bad




    model_num = 713600  ### good but not very good story



class isac_adaptive_character__intersection(ModelPath):
    method = 'independentsac_v0'

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-14-10:25:41----ray_isac_adaptive_character__intersection/saved_models_method'
    
    model_num = 296200  ### good ?
    model_num = 296400  ### bad
    model_num = 296600  ### bad
    model_num = 296800  ### bad
    

    
    model_num = 343800  ### good but not good story, for qualitative
    model_num = 344000  ### bad
    model_num = 344200  ### bad
    model_num = 344400  ### good ?
    model_num = 344600  ### good ?
    model_num = 344800  ### bad
    model_num = 345000  ### bad

    model_num = 400800  ### good but not good story, for qualitative
    model_num = 401000  ### good ?
    model_num = 401200  ### good ?
    model_num = 401400  ### bad
    model_num = 401600  ### bad
    model_num = 401800  ### bad
    model_num = 402000  ### bad
    model_num = 402200  ### bad
    model_num = 402400  ### bad
    model_num = 402600  ### good ?
    model_num = 402800  ### bad
    model_num = 403000  ### bad

    model_num = 422200  ### bad
    model_num = 422400  ### good !!!, maybe qualitative
    model_num = 422600  ### good ?
    model_num = 422800  ### bad
    model_num = 423000  ### just so so
    model_num = 423200  ### just so so
    model_num = 423400  ### just so so
    model_num = 423600  ### just so so


    ### final tmp
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-14-10:25:41----ray_isac_adaptive_character__intersection/saved_models_method'
    model_num = 401200  ### good ?
    
    model_num = 422400  ### for qualitative
    model_num = 422600  ### final

    ### final
    model_dir = '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-14-10:25:41----ray_isac_adaptive_character__intersection/saved_models_method'
    model_num = 422600



class isac_adaptive_character__merge(ModelPath):
    method = 'independentsac_v0'

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method'
    model_num = 698600  ### good
    model_num = 866200  ### good

    model_num = 1027000  ### bad
    model_num = 1026800  ### bad
    model_num = 1026600  ### bad
    model_num = 1027200  ### bad
    model_num = 1027400  ### bad
    model_num = 1027600  ### bad
    model_num = 1027800  ### bad
    model_num = 1028000  ### bad


    ### final
    model_dir = '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method'
    model_num = 866200  ### good



class isac_adaptive_character__roundabout(ModelPath):
    method = 'independentsac_v0'

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method'
    
    model_num = 865600  ### just so so
    model_num = 865800  ### bad
    model_num = 866000  ### just so so
    model_num = 866200  ### just so so
    model_num = 866400  ### bad
    model_num = 866600  ### bad
    model_num = 866800  ### bad
    model_num = 867000  ### bad

    model_num = 868000

    ### final ?
    model_num = 868400
    model_num = 869400


    ### final
    model_dir = '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method'
    model_num = 869400











class isac_adaptive_character__bottleneck__qualitive(ModelPath):
    method = 'independentsac_v0'

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-03-21:01:30----ray_isac_adaptive_character__bottleneck/saved_models_method'
    model_num = 945800  ### good, low off road


class isac_adaptive_character__intersection__qualitive(ModelPath):
    method = 'independentsac_v0'

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-07-23:20:15----ray_isac_adaptive_character__intersection/saved_models_method'
    model_num = 527800  ### good, low failure









class isac_adaptive_character__multi_scenario(ModelPath):
    method = 'independentsac_v0'

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-05-17:43:01----ray_isac_adaptive_character__multi_scenario/saved_models_method'
    model_num = 508000  ### bad
    model_num = 508200  ### bad
    model_num = 508400  ### just so so
    model_num = 508600  ### bad
    model_num = 508800  ### bad
    model_num = 509400  ### bad
    

    model_num = 508000  ### bad



    # model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-10-23:40:37----ray_isac_adaptive_character__multi_scenario--random-0.25/saved_models_method'
    # model_num = 202400  ### ! tmp

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-10-23:26:37----ray_isac_adaptive_character__multi_scenario--buffer-0.8/saved_models_method'
    model_num = 206600  ### ! tmp
    model_num = 289600  ### ! tmp

##################recognition

class isac__bottleneck__adaptive(ModelPath):
    method = 'IndependentSAC_v0'

    #2022.10.10 test in 116 server
    model_dir = '~/github/zdk/recognition-rl/models/IndependentSAC_v0-EnvInteractiveMultiAgent/2022-09-11-15:19:29----ray_isac_adaptive_character__multi_scenario--buffer-rate-0.2/saved_models_method'
    model_num = 865800
    #2022/12/21
    # model_dir = '~/github/zdk/recognition-rl/models/origin_no_history_bottleneck'
    # model_num = 445600

class RILMthM__bottleneck(ModelPath):

    method = 'IndependentSAC_recog'
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_recog-EnvInteractiveMultiAgent/2023-02-03-13:19:11----Nothing--RILMthM/saved_models_method'
    
    model_num = 140000
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSAC_recog-EnvInteractiveMultiAgent/2023-02-13-10:57:15----Nothing--RILMthM/saved_models_method'

    model_num = 200000
    model_num = 204000
    model_num = 202000
    model_num = 199000
    model_num = 198000
    model_num = 197000
    model_num = 196000
    model_num = 195000
    model_num = 194000
    model_num = 203000
class RILEnvM__bottleneck(ModelPath):
    method = 'RecogV2'
    model_dir = ''

class IL__bottleneck(ModelPath):
    method = 'IndependentSACsupervise'
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveMultiAgent/2023-02-06-10:09:14----Nothing--supervise-MultiAgent/saved_models_method'
    model_num = 78000

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveMultiAgent/2023-02-09-20:29:40----Nothing--IL-close-loop/saved_models_method'
    model_num = 700000
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveMultiAgent/2023-02-13-20:01:32----Nothing--IL-close-loop/saved_models_method'
    model_num = 790000
    model_num = 780000
    model_num = 770000
    model_num = 760000
    model_num = 750000
    model_num = 740000
    model_num = 730000
    model_num = 190000
    model_num = 220000
class IL_offline__bottleneck(ModelPath):

    method = 'IndependentSACsupervise'

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveMultiAgent/2023-02-06-10:09:14----Nothing--supervise-MultiAgent/saved_models_method'
    model_num = 78000
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveMultiAgent/2023-02-14-16:10:01----Nothing--IL-open-loop/saved_models_method'
    model_num = 990000
    model_num = 720000
    model_num = 220000
    model_num = 240000

    #hr = 1 sampled  116
    # model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveMultiAgent/2023-02-19-14:13:49----Nothing--IL-open-loop/saved_models_method'
    # model_num = 145000

    # #hr = 5 sampled  115
    # model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveMultiAgent/2023-02-19-14:13:49----Nothing--IL-open-loop/saved_models_method'
    # model_num = 145000
    #hr = 10 sampled 115
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveMultiAgent/2023-02-19-11:13:06----Nothing--IL-open-loop/saved_models_method'
    model_num = 145000

    # #hr = 15 sampled 116
    # model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveMultiAgent/2023-02-19-11:40:50----Nothing--IL-open-loop/saved_models_method'
    # model_num = 145000

    # 115 hr = 10 without map
    # model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveMultiAgent/2023-02-19-22:30:38----Nothing--IL-open-loop_womap/saved_models_method'
    # model_num = 145000

    # 115 hr = 10 without attention (problem)
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveMultiAgent/2023-02-20-01:52:15----Nothing--IL-open-loop_woattn/saved_models_method'
    model_num = 145000

    # 116 hr = 1 
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveMultiAgent/2023-02-20-01:44:04----Nothing--IL_open_loop_hr1__merge/saved_models_method'
    model_num = 145000

    # 116 hr = 5
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveMultiAgent/2023-02-20-01:44:04----Nothing--IL_open_loop_hr1__merge/saved_models_method'
    model_num = 145000

    # 116 hr = 10
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveMultiAgent/2023-02-20-01:44:04----Nothing--IL_open_loop_hr1__merge/saved_models_method'
    model_num = 145000

    # 116 hr = 15
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveMultiAgent/2023-02-20-01:44:04----Nothing--IL_open_loop_hr1__merge/saved_models_method'
    model_num = 145000
class IL_offline__merge(ModelPath):

    method = 'IndependentSACsupervise'

    model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveMultiAgent/2023-02-06-10:09:14----Nothing--supervise-MultiAgent/saved_models_method'
    model_num = 78000
    model_dir = '~/github/zdk/recognition-rl/results/IndependentSACsupervise-EnvInteractiveMultiAgent/2023-02-14-16:10:01----Nothing--IL-open-loop/saved_models_method'
