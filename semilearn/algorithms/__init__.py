# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

<<<<<<< HEAD

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
# from .mpl import MPL
from .fullysupervised import FullySupervised
from .comatch import CoMatch
from .simmatch import SimMatch
from .adamatch import AdaMatch
from .refixmatch import ReFixMatch
from .sequencematch import SequenceMatch

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
    # 'mpl': MPL
    'sequencematch': SequenceMatch,
}
=======
from semilearn.core.utils import ALGORITHMS
name2alg = ALGORITHMS
>>>>>>> c9709aa50394658aa4b2666a34c6179d22b18033

def get_algorithm(args, net_builder, tb_log, logger):
    if args.algorithm in ALGORITHMS:
        alg = ALGORITHMS[args.algorithm]( # name2alg[args.algorithm](
            args=args,
            net_builder=net_builder,
            tb_log=tb_log,
            logger=logger
        )
        return alg
    else:
        raise KeyError(f'Unknown algorithm: {str(args.algorithm)}')



