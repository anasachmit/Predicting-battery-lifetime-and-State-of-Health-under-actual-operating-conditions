# =============================================================================
#  ONE SWITCH: which model does run_model.py use?
#
#  Pour revenir au modele parametrique L3 (v3), commenter la ligne PFN et
#  decommenter la ligne model_template : c'est un repli en UNE ligne.
# =============================================================================
# from .model_example import ExampleModel as ActiveModel    # reference baseline
# from .model_template import MyModel as ActiveModel        # L3 parametrique (v3)
from .model_pfn import BatteryPFN as ActiveModel            # <-- BatteryPFN
