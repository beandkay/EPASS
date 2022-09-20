# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from .fixmatch import FixMatch
from .flexmatch import FlexMatch
from .pimodel import PiModel
from .meanteacher import MeanTeacher
from .pseudolabel import PseudoLabel
from .uda import UDA
from .mixmatch import MixMatch
from .vat import VAT
from .remixmatch import ReMixMatch
from .crmatch import CRMatch
from .dash import Dash
from .mpl import MPL
from .fullysupervised import FullySupervised
from .comatch import CoMatch
from .simmatch import SimMatch
from .adamatch import AdaMatch
from .refixmatch import ReFixMatch
from .sequencematch import SequenceMatch
from .somematch import SomeMatch
from .cagul import CAGUL

# if any new alg., please append the dict
name2alg = {
    'fullysupervised': FullySupervised,
    'supervised': FullySupervised,
    'fixmatch': FixMatch,
    'refixmatch': ReFixMatch,
    'flexmatch': FlexMatch,
    'adamatch': AdaMatch,
    'pimodel': PiModel,
    'meanteacher': MeanTeacher,
    'pseudolabel': PseudoLabel,
    'uda': UDA,
    'vat': VAT,
    'mixmatch': MixMatch,
    'remixmatch': ReMixMatch,
    'crmatch': CRMatch,
    'comatch': CoMatch,
    'simmatch': SimMatch,
    'dash': Dash,
    'mpl': MPL,
    'sequencematch': SequenceMatch,
    'somematch': SomeMatch,
    'cagul': CAGUL
}


def get_algorithm(args, net_builder, tb_log, logger):
    try:
        model = name2alg[args.algorithm](
            args=args,
            net_builder=net_builder,
            tb_log=tb_log,
            logger=logger
        )
        return model
    except KeyError as e:
        print(f'Unknown algorithm: {str(e)}')



